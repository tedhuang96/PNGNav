#!/home/zhe/miniconda3/envs/pngenv/bin/python
from os.path import join

import rospy
import numpy as np

from png_navigation.configs.rrt_star_config import Config
from png_navigation.path_planning_classes.rrt_env_2d import Env
from png_navigation.path_planning_classes.irrt_star_2d import get_path_planner

from png_navigation.srv import SetEnv, SetEnvResponse
from png_navigation.srv import GetGlobalPlan, GetGlobalPlanResponse


def get_fake_env():
    env_dict = {
        'x_range': [0,10],
        'y_range': [0,10],
        'circle_obstacles': [],
        'rectangle_obstacles': [],
    }
    return Env(env_dict)

def get_fake_problem():
    problem = {
        'x_start': [0,0],
        'x_goal': [1,1],
        'search_radius': 10,
        'env': get_fake_env(),
    }
    return problem

class IRRTStarNode:
    def __init__(
        self,
    ):
        self.config = Config()
        self.planner = get_path_planner(
            self.config.path_planner_args,
            get_fake_problem(),
        )
        rospy.Service('png_navigation/set_env_2d', SetEnv, self.set_env)
        rospy.Service('png_navigation/get_global_plan', GetGlobalPlan, self.get_global_plan)
    
    def get_global_plan(self, request):
        self.planner.reset_robot(
            x_start=request.plan_request.start,
            x_goal=request.plan_request.goal,
            env=None,
            search_radius=request.plan_request.search_radius,
            max_time=request.plan_request.max_time,
        )
        # * clearance and max_iterations from plan_request are redundant and not used here.
        path = self.planner.planning_robot()
        if len(path) == 0:
            is_solved = False
            path = []
            return GetGlobalPlanResponse(is_solved, path, request.plan_request)
        else:
            is_solved = True
            path = path.flatten()
            return GetGlobalPlanResponse(is_solved, path, request.plan_request)

    def set_env(self, request):
        if len(request.request_env.circle_obstacles)>0:
            circle_obstacles = np.array(request.request_env.circle_obstacles).reshape(-1,3)
        else:
            circle_obstacles = []
        if len(request.request_env.rectangle_obstacles)>0:
            rectangle_obstacles = np.array(request.request_env.rectangle_obstacles).reshape(-1,3)
        else:
            rectangle_obstacles = []       
        env_dict = {
            'x_range': request.request_env.x_range,
            'y_range': request.request_env.y_range,
            'circle_obstacles': circle_obstacles,
            'rectangle_obstacles': rectangle_obstacles,
        }
        self.planner.reset_env_robot(Env(env_dict))
        rospy.loginfo("Environment is set.")
        is_set = True
        return SetEnvResponse(is_set)

if __name__ == '__main__':
    try:
        rospy.init_node('png_navigation_irrt_star_node', anonymous=True)
        irrtsn = IRRTStarNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

