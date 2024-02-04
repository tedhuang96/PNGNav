
class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):

    robot_config = BaseConfig()
    robot_config.model_name = 'turtlebot2' # or 'turtlebot3_waffle_pi'
    robot_config.length = 0.281 # meter for robot_config
    robot_config.width = 0.306
    robot_config.height = 0.141
    robot_config.wheel_distance = 0.287
    robot_config.clearance_radius = 0.25 # tb2i with additional inflation radius. tb2i: 0.18; tb3: 0.220 

    path_planner_args = BaseConfig()
    path_planner_args.step_len = 0.5
    path_planner_args.search_radius = 3. # computed using free space approximation.
    path_planner_args.iter_max = 50000
    path_planner_args.clearance = robot_config.clearance_radius
    path_planner_args.max_time = 2 # total planning time. unit: second
    path_planner_args.pc_update_cost_ratio = 0.95
    path_planner_args.pc_sample_rate = 0.5

    ros_config = BaseConfig()
    ros_config.nav_goal_topic = '/move_base_simple/goal'
    ros_config.robot_frame = 'base_footprint'

    png_config = BaseConfig()
    png_config.pc_n_points = 2048
    png_config.pc_over_sample_scale = 5
    png_config.clearance = robot_config.clearance_radius
    png_config.step_len = 0.3
    png_config.connect_max_trial_attempts = 3