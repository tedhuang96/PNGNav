import rospy
import numpy as np

from png_navigation.path_planning_utils.rrt_env import Env
from png_navigation.path_planning_classes.rrt_base_2d import RRTBase2D
from png_navigation.path_planning_classes.nirrt_star_png_2d import NIRRTStarPNG2D
from png_navigation.path_planning_classes.rrt_visualizer_2d import NIRRTStarVisualizer

from std_msgs.msg import String, Int32, Float64MultiArray

from png_navigation.msg import NIRRTWrapperMsg

class NIRRTStarPNGC2D(NIRRTStarPNG2D):
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        search_radius,
        iter_max,
        env_dict,
        clearance,
        pc_n_points,
        pc_over_sample_scale,
        pc_sample_rate,
        pc_update_cost_ratio,
        connect_max_trial_attempts,
    ):
        RRTBase2D.__init__(
            self,
            x_start,
            x_goal,
            step_len,
            search_radius,
            iter_max,
            Env(env_dict),
            clearance,
            "NIRRT*-PNG(C) 2D",
        )
        self.pc_n_points = pc_n_points # * number of points in pc
        self.pc_over_sample_scale = pc_over_sample_scale
        self.pc_sample_rate = pc_sample_rate
        self.pc_neighbor_radius = self.step_len
        self.pc_update_cost_ratio = pc_update_cost_ratio
        self.connect_max_trial_attempts = connect_max_trial_attempts
        
        self.env_dict = env_dict
        self.path_solutions = [] # * a list of valid goal parent vertex indices
        self.visualizer = NIRRTStarVisualizer(self.x_start, self.x_goal, self.env)

        self.planning_start_end_pub = rospy.Publisher('planning_start_end', String, queue_size=10)
        self.path_len_pub = rospy.Publisher('path_len', Float64MultiArray, queue_size=10)
        self.neural_wrapper_pub = rospy.Publisher('wrapper_input', NIRRTWrapperMsg, queue_size=10)
        self.time_pub = rospy.Publisher('time_record', String, queue_size=10)
        self.current_path_len = np.inf
        self.path_point_cloud_pred = None
        self.cmax = np.inf
        self.cmin = 0.  # invalid
        rospy.sleep(1)
        rospy.Subscriber('random_timer', Int32, self.random_timer_callback)
        rospy.Subscriber('wrapper_output', Float64MultiArray, self.pc_callback)

    def reset(
        self,
        x_start,
        x_goal,
        env_dict,
        search_radius,
    ):
        RRTBase2D.reset(
            self,
            x_start,
            x_goal,
            Env(env_dict),
            search_radius,
        )
        self.env_dict = env_dict
        self.path_solutions = [] # * a list of valid goal parent vertex indices
        self.visualizer = NIRRTStarVisualizer(self.x_start, self.x_goal, self.env)
        self.current_path_len = np.inf
        self.path_point_cloud_pred = None
        self.cmax = np.inf
        self.cmin = 0.  # invalid

    def visualize(self, x_center, c_best, start_goal_straightline_dist, theta, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = "nirrt*(c) 2D, iteration " + str(self.iter_max)
        if img_filename is None:
            img_filename="nirrt*_c_2d_example.png"
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

def get_path_planner(
    args,
    problem,
    neural_wrapper,
):
    return NIRRTStarPNGC2D(
        problem['x_start'],
        problem['x_goal'],
        args.step_len,
        problem['search_radius'],
        args.iter_max,
        problem['env_dict'],
        neural_wrapper,
        problem['binary_mask'],
        args.clearance,
        args.pc_n_points,
        args.pc_over_sample_scale,
        args.pc_sample_rate,
        args.pc_update_cost_ratio,
        args.connect_max_trial_attempts,
    )