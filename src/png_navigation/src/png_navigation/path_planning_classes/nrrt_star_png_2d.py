import time

import rospy
import numpy as np

from png_navigation.msg import NIRRTWrapperMsg
from png_navigation.path_planning_classes.rrt_base_2d import RRTBase2D
from png_navigation.path_planning_classes.rrt_star_2d import RRTStar2D
from png_navigation.path_planning_classes.rrt_visualizer_2d import NRRTStarPNGVisualizer

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Int32, Float64MultiArray

class NRRTStarPNG2D(RRTStar2D):
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
            "NRRT*-PNG 2D",
        )
        self.pc_sample_rate = pc_sample_rate
        self.pc_neighbor_radius = self.step_len
        self.visualizer = NRRTStarPNGVisualizer(self.x_start, self.x_goal, self.env)
        self.planning_start_end_pub = rospy.Publisher('planning_start_end', String, queue_size=10)
        self.path_len_pub = rospy.Publisher('path_len', Float64MultiArray, queue_size=10)
        self.time_pub = rospy.Publisher('time_record', String, queue_size=10)
        self.neural_wrapper_pub = rospy.Publisher('wrapper_input', NIRRTWrapperMsg, queue_size=10)
        self.tree_pub = rospy.Publisher('tree', MarkerArray, queue_size=10)
        self.current_path = None
        self.path_point_cloud_pred = None
        rospy.sleep(1)
        rospy.Subscriber('random_timer', Int32, self.random_timer_callback)
        rospy.Subscriber('wrapper_output', Float64MultiArray, self.pc_callback)

    def random_timer_callback(self, data):
        time_after_initial = data.data*0.5
        path_len = self.get_path_len(self.current_path)
        msg = Float64MultiArray()
        msg.data = [time_after_initial, path_len]
        self.path_len_pub.publish(msg)

    def pc_callback(self, data):
        self.path_point_cloud_pred = np.array(data.data).reshape((-1, 2))
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
        self.current_path = None
        self.visualizer = NRRTStarPNGVisualizer(self.x_start, self.x_goal, self.env)
        self.path_point_cloud_pred = None

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
        self.current_path = None
        self.visualizer = NRRTStarPNGVisualizer(self.x_start, self.x_goal, self.env)
        self.path_point_cloud_pred = None

    def init_pc(self):
        self.update_point_cloud()

    def planning(self, visualize=False):
        self.init_pc()
        RRTStar2D.planning(self, visualize)

    def planning_robot(self, visualize=False):
        start_time = time.time()
        self.init_pc()
        path = RRTStar2D.planning_robot(self, visualize, start_time)
        tree_msg = MarkerArray()
        marker_id = 0
        for vertex_index, vertex_parent_index in enumerate(self.vertex_parents[:self.num_vertices]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = marker_id  # Each marker must have a unique ID
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02#2# 0.005
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
        return path

    def generate_random_node(self):
        if np.random.random() < self.pc_sample_rate:
            return self.SamplePointCloud()
        else:
            return self.SampleFree()

    def SamplePointCloud(self):
        if self.path_point_cloud_pred is not None:
            # print("have pc ready now")
            return self.path_point_cloud_pred[np.random.randint(0,len(self.path_point_cloud_pred))]
        else:
            # print("does not have pc ready yet")
            return self.SampleFree()
        
    def visualize(self, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = "nrrt*-png 2D, iteration " + str(self.iter_max)
        if img_filename is None:
            img_filename = "nrrt*_png_2d_example.png"
        self.visualizer.animation(
            self.vertices[:self.num_vertices],
            self.vertex_parents[:self.num_vertices],
            self.path,
            figure_title,
            animation=False,
            img_filename=img_filename)
        
    def update_point_cloud(self):
        # print("Run neural wrapper")
        if self.pc_sample_rate == 0:
            self.path_point_cloud_pred = None
            self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)
            return
        msg = NIRRTWrapperMsg()
        msg.x_start = list(self.x_start)
        msg.x_goal = list(self.x_goal)
        msg.c_max = 0 # dummy
        msg.c_min = 0 # dummy
        self.neural_wrapper_pub.publish(msg)

    def planning_block_gap(
        self,
        path_len_threshold,
        max_time_threshold=120,
    ):
        start_time = time.time()
        self.init_pc()
        return RRTStar2D.planning_block_gap(self, path_len_threshold, max_time_threshold=max_time_threshold, start_time=start_time)

    def planning_random(
        self,
        env_idx,
        time_after_initial=62,
    ):
        total_time_start = time.time()
        self.init_pc()
        return RRTStar2D.planning_random(self, env_idx, time_after_initial, total_time_start)


def get_path_planner(
    args,
    problem,
):
    return NRRTStarPNG2D(
        problem['x_start'],
        problem['x_goal'],
        args.step_len,
        problem['search_radius'],
        args.iter_max,
        problem['env'],
        args.clearance,
        args.pc_sample_rate,
    )