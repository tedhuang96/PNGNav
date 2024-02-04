import time

import rospy
import numpy as np

from png_navigation.path_planning_classes.rrt_base_2d import RRTBase2D
from png_navigation.path_planning_classes.irrt_star_2d import IRRTStar2D
from png_navigation.path_planning_classes.rrt_visualizer_2d import NIRRTStarVisualizer

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Int32, Float64MultiArray

from png_navigation.msg import NIRRTWrapperMsg


class NIRRTStarPNG2D(IRRTStar2D):
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        search_radius,
        iter_max,
        env,
        clearance,
        pc_sample_rate,
        pc_update_cost_ratio,
    ):
        RRTBase2D.__init__(
            self,
            x_start,
            x_goal,
            step_len,
            search_radius,
            iter_max,
            env,
            clearance,
            "NIRRT*-PNG 2D",
        )
        # self.png_wrapper = png_wrapper
        # self.binary_mask = binary_mask
        self.pc_sample_rate = pc_sample_rate
        self.pc_neighbor_radius = self.step_len
        self.pc_update_cost_ratio = pc_update_cost_ratio
        
        self.path_solutions = [] # * a list of valid goal parent vertex indices
        self.visualizer = NIRRTStarVisualizer(self.x_start, self.x_goal, self.env)

        self.planning_start_end_pub = rospy.Publisher('planning_start_end', String, queue_size=10)
        self.path_len_pub = rospy.Publisher('path_len', Float64MultiArray, queue_size=10)
        self.neural_wrapper_pub = rospy.Publisher('wrapper_input', NIRRTWrapperMsg, queue_size=10)
        self.time_pub = rospy.Publisher('time_record', String, queue_size=10)

        self.tree_pub = rospy.Publisher('tree', MarkerArray, queue_size=10)

        self.current_path_len = np.inf
        self.path_point_cloud_pred = None
        self.cmax = np.inf
        self.cmin = 0.  # invalid
        rospy.sleep(1)
        rospy.Subscriber('random_timer', Int32, self.random_timer_callback)
        rospy.Subscriber('wrapper_output', Float64MultiArray, self.pc_callback)

    def random_timer_callback(self, data):
        time_after_initial = data.data*0.5
        msg = Float64MultiArray()
        msg.data = [time_after_initial, self.current_path_len]
        self.path_len_pub.publish(msg)

    def pc_callback(self, data):
        self.path_point_cloud_pred = np.array(data.data).reshape((-1, 2)) # (n, 2)
        if self.cmax < np.inf:
            in_flag = np.linalg.norm(self.path_point_cloud_pred-self.x_start, axis=1)+\
                np.linalg.norm(self.path_point_cloud_pred-self.x_goal, axis=1)<self.cmax # (n)
            if len(self.path_point_cloud_pred)==len(in_flag): # conflict resolved
                self.path_point_cloud_pred = self.path_point_cloud_pred[in_flag]  
        self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)

    def reset(
        self,
        x_start,
        x_goal,
        env,
        search_radius,
    ):
        RRTBase2D.reset(
            self,
            x_start,
            x_goal,
            env,
            search_radius,
        )
        self.path_solutions = [] # * a list of valid goal parent vertex indices
        self.visualizer = NIRRTStarVisualizer(self.x_start, self.x_goal, self.env)
        self.current_path_len = np.inf
        self.path_point_cloud_pred = None
        self.cmax = np.inf
        self.cmin = 0.  # invalid

    def reset_robot(
        self,
        x_start,
        x_goal,
        env,
        search_radius,
        max_time,
    ):
        if self.num_vertices > 1:
            tree_msg = MarkerArray()
            marker_id = 0
            for vertex_index, vertex_parent_index in enumerate(self.vertex_parents[:self.num_vertices]):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.id = marker_id  # Each marker must have a unique ID
                marker.action = Marker.DELETE
                tree_msg.markers.append(marker)
                marker_id += 1
            self.tree_pub.publish(tree_msg)
        
        RRTBase2D.reset_robot(
            self,
            x_start,
            x_goal,
            env,
            search_radius,
            max_time,
        )
        self.path_solutions = [] # * a list of valid goal parent vertex indices
        self.visualizer = NIRRTStarVisualizer(self.x_start, self.x_goal, self.env)
        self.current_path_len = np.inf
        self.path_point_cloud_pred = None
        self.cmax = np.inf
        self.cmin = 0.  # invalid

    def init_pc(self):
        self.update_point_cloud(
            cmax=np.inf,
            cmin=0., # invalid
        )

    def planning(
        self,
        visualize=False,
    ):
        theta, start_goal_straightline_dist, x_center, C = self.init()
        self.init_pc() # * nirrt*
        c_best = np.inf
        c_update = c_best # * nirrt*
        self.cmax = c_best
        self.cmin = start_goal_straightline_dist
        for k in range(self.iter_max):
            if k % 1000 == 0:
                print(k)
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
                self.cmax = c_best
            node_rand, c_update = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C, c_update) # * nirrt*
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)
                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)
        if len(self.path_solutions)>0:
            c_best, x_best = self.find_best_path_solution()
            self.cmax = c_best
            self.path = self.extract_path(x_best)
        else:
            self.path = []
        if visualize:
            self.visualize(x_center, c_best, start_goal_straightline_dist, theta)


    def planning_robot(
        self,
        visualize=False,
        start_time=None,
    ):
        k = 0
        if not start_time:
            start_time = time.time()
        theta, start_goal_straightline_dist, x_center, C = self.init()
        self.init_pc() # * nirrt*
        c_best = np.inf
        c_update = c_best # * nirrt*
        self.cmax = c_best
        self.cmin = start_goal_straightline_dist
        while time.time()-start_time < self.max_time_robot:
            if k % 1000 == 0:
                print(k)
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
                self.cmax = c_best
            node_rand, c_update = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C, c_update) # * nirrt*
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)
                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)
            k += 1
        if len(self.path_solutions)>0:
            c_best, x_best = self.find_best_path_solution()
            self.cmax = c_best
            self.path = self.extract_path(x_best)
        else:
            self.path = []
        if visualize:
            self.visualize(x_center, c_best, start_goal_straightline_dist, theta)

        tree_msg = MarkerArray()
        marker_id = 0
        for vertex_index, vertex_parent_index in enumerate(self.vertex_parents[:self.num_vertices]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = marker_id  # Each marker must have a unique ID
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            point1 = Point()
            point1.x = self.vertices[:self.num_vertices][vertex_index,0]
            point1.y = self.vertices[:self.num_vertices][vertex_index,1]
            point1.z = 0
            point2 = Point()
            point2.x = self.vertices[:self.num_vertices][vertex_parent_index,0]
            point2.y = self.vertices[:self.num_vertices][vertex_parent_index,1]
            point2.z = 0
            marker.points.append(point1)
            marker.points.append(point2)
            tree_msg.markers.append(marker)
            marker_id += 1
        self.tree_pub.publish(tree_msg)
        
        return self.path


    def generate_random_node(
        self,
        c_curr,
        c_min,
        x_center,
        C,
        c_update,
    ):
        '''
        - outputs
            - node_rand: np (2,)
            - c_update: scalar
        '''
        # * tested that np.inf < alpha*np.inf is False, alpha in (0,1]
        if c_curr < self.pc_update_cost_ratio*c_update:
            self.update_point_cloud(c_curr, c_min)
            c_update = c_curr
        if np.random.random() < self.pc_sample_rate:
            return self.SamplePointCloud(c_curr, c_min, x_center, C), c_update
        else:
            if c_curr < np.inf:
                return self.SampleInformedSubset(
                    c_curr,
                    c_min,
                    x_center,
                    C,
                ), c_update
            else:
                return self.SampleFree(), c_update

    def SamplePointCloud(self, c_curr, c_min, x_center, C):
        if self.path_point_cloud_pred is not None and self.path_point_cloud_pred.shape[0]>0:
            # print("have pc ready now")
            return self.path_point_cloud_pred[np.random.randint(0,len(self.path_point_cloud_pred))]
        else:
            # print("does not have pc ready yet")
            if c_curr < np.inf:
                return self.SampleInformedSubset(
                    c_curr,
                    c_min,
                    x_center,
                    C,
                )
            else:
                return self.SampleFree()


    def update_point_cloud(
        self,
        cmax,
        cmin,
    ):
        # print("Using update_point_cloud from nirrt*-png")
        if self.pc_sample_rate == 0:
            self.path_point_cloud_pred = None
            self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)
            return
        msg = NIRRTWrapperMsg()
        msg.x_start = list(self.x_start)
        msg.x_goal = list(self.x_goal)
        msg.c_max = cmax
        msg.c_min = cmin
        self.neural_wrapper_pub.publish(msg)

    def visualize(self, x_center, c_best, start_goal_straightline_dist, theta, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = "nirrt* 2D, iteration " + str(self.iter_max)
        if img_filename is None:
            img_filename="nirrt*_2d_example.png"
        self.visualizer.animation(
            self.vertices[:self.num_vertices],
            self.vertex_parents[:self.num_vertices],
            self.path,
            figure_title,
            x_center,
            c_best,
            start_goal_straightline_dist,
            theta,
            img_filename=img_filename,
        )


    def planning_block_gap(
        self,
        path_len_threshold,
        max_time_threshold=120,
    ):
        start_time = time.time()
        path_len_list = []
        theta, start_goal_straightline_dist, x_center, C = self.init()
        self.init_pc() # * nirrt*
        c_best = np.inf
        c_update = c_best # * nirrt*
        self.cmax = c_best
        self.cmin = start_goal_straightline_dist
        better_than_path_len_threshold = False
        for k in range(self.iter_max):
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
                self.cmax = c_best
            iter_time = time.time()
            if iter_time-start_time>max_time_threshold:
                break
            path_len_list.append([c_best, iter_time-start_time])
            if k % 1000 == 0:
                print("{0}/{1} - current: {2:.2f}, threshold: {3:.2f}".format(\
                    k, self.iter_max, c_best, path_len_threshold)) #* not k+1, because we are not getting c_best after iteration is done
            if c_best < path_len_threshold:
                better_than_path_len_threshold = True
                break
            node_rand, c_update = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C, c_update) # * nirrt*
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)
                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)
        path_len_list = path_len_list[1:] # * the first one is the initialized c_best before iteration
        if better_than_path_len_threshold:
            return path_len_list
        # * path cost for the last iteration
        if len(self.path_solutions)>0:
            c_best, x_best = self.find_best_path_solution()
            self.cmax = c_best
        iter_time = time.time()
        if iter_time-start_time<max_time_threshold:
            path_len_list.append([c_best, iter_time-start_time])
        # * len(path_len_list)==self.iter_max
        print("{0}/{1} - current: {2:.2f}, threshold: {3:.2f}".format(\
            len(path_len_list), self.iter_max, c_best, path_len_threshold)) #* not k+1, because we are not getting c_best after iteration is done
        return path_len_list

    def planning_random(
        self,
        env_idx,
        time_after_initial=62,
    ):
        total_time_start = time.time()
        # * set 62 to add a little buffer for 60 sec
        path_len_list = []
        theta, start_goal_straightline_dist, x_center, C = self.init()
        self.init_pc() # * nirrt*
        c_best = np.inf
        c_update = c_best # * nirrt*
        self.cmax = c_best
        self.cmin = start_goal_straightline_dist
        better_than_inf = False
        total_iter_count = 0
        for k in range(self.iter_max):
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
                self.cmax = c_best
            path_len_list.append(c_best)
            if k % 1000 == 0:
                if c_best == np.inf:
                    print("{0}/{1} - current: inf".format(k, self.iter_max)) #* not k+1, because we are not getting c_best after iteration is done
            if c_best < np.inf:
                better_than_inf = True
                print("{0}/{1} - current: {2:.2f}".format(k, self.iter_max, c_best))
                break
            node_rand, c_update = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C, c_update) # * nirrt*
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)
                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)
            total_iter_count += 1
            if total_iter_count % 500 == 0:
                self.time_pub.publish("iteration count: {0}, time: {1}".format(total_iter_count, time.time()-total_time_start))
        path_len_list = path_len_list[1:] # * the first one is the initialized c_best before iteration
        if better_than_inf:
            initial_path_len = path_len_list[-1]
        else:
            # * path cost for the last iteration
            if len(self.path_solutions)>0:
                c_best, x_best = self.find_best_path_solution()
                self.cmax = c_best
            path_len_list.append(c_best)
            initial_path_len = path_len_list[-1]
            if initial_path_len == np.inf:
                # * fail to find initial path solution
                # return path_len_list
                self.planning_start_end_pub.publish("start_"+str(env_idx))
                self.planning_start_end_pub.publish("end_"+str(env_idx))
                return
        self.current_path_len = initial_path_len
        self.planning_start_end_pub.publish("start_"+str(env_idx))
        start_time = time.time()
        path_len_list = path_len_list[:-1] # * for loop below will add initial_path_len to path_len_list
        # * iteration after finding initial solution
        k = 0
        while time.time()-start_time < time_after_initial:
            c_best, x_best = self.find_best_path_solution() # * there must be path solutions
            self.cmax = c_best
            self.current_path_len = c_best
            path_len_list.append(c_best)
            if k % 1000 == 0:
                print("iter {0}/time {1:.2f} - current: {2:.2f}, initial: {3:.2f}, cmin: {4:.2f}".format(\
                    k, time.time()-start_time, c_best, initial_path_len, start_goal_straightline_dist))
            node_rand, c_update = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C, c_update) # * nirrt*
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)
            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new-node_nearest)<1e-8:
                    # * do not create a new node if it is actually the same point
                    node_new = node_nearest
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index)+self.Line(node_nearest, node_new)
                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices)>0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
                if self.InGoalRegion(node_new):
                    self.path_solutions.append(node_new_index)
            k += 1
            total_iter_count += 1
            if total_iter_count % 500 == 0:
                self.time_pub.publish("iteration count: {0}, time: {1}".format(total_iter_count, time.time()-total_time_start))

        self.planning_start_end_pub.publish("end_"+str(env_idx))
        # * path cost for the last iteration
        c_best, x_best = self.find_best_path_solution() # * there must be path solutions
        self.cmax = c_best
        self.current_path_len = c_best
        path_len_list.append(c_best)
        print("iter {0}/time {1:.2f} - current: {2:.2f}, initial: {3:.2f}".format(\
            k, time.time()-start_time, c_best, initial_path_len))

def get_path_planner(
    args,
    problem,
):
    return NIRRTStarPNG2D(
        problem['x_start'],
        problem['x_goal'],
        args.step_len,
        problem['search_radius'],
        args.iter_max,
        problem['env'],
        args.clearance,
        args.pc_sample_rate,
        args.pc_update_cost_ratio,
    )
    