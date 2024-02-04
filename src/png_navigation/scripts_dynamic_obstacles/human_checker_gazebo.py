#!/usr/bin/python3.8
import math

import tf
import rospy
import numpy as np
from tf.transformations import euler_from_quaternion

from png_navigation.path_planning_classes.rrt_env_2d import Env
from png_navigation.path_planning_classes.rrt_utils_2d import Utils

from std_msgs.msg import Bool
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped
from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

from png_navigation.srv import GlobalReplan
from png_navigation.srv import StopRobot, ResumeRobot


def normalize_angle(angle):
    normalized_angle = math.fmod(angle + math.pi, 2 * math.pi)
    if normalized_angle < 0:
        normalized_angle += 2*math.pi
    return normalized_angle - math.pi

def get_fake_env():
    env_dict = {
        'x_range': [-100,100],
        'y_range': [-100,100],
        'circle_obstacles': [],
        'rectangle_obstacles': [],
    }
    return Env(env_dict)

class HumanChecker:
    def __init__(
        self,
        robot_frame='base_footprint',
        human_detection_radius=0.5,
        human_clearance=0.1,
    ):
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'map'
        self.base_frame = robot_frame

        self.human_detection_radius = human_detection_radius # * meter
        self.human_clearance = human_clearance
        self.humans = np.array([])
        self.interactive_humans = np.array([])

        self.global_path = np.array([])
        self.path_waypoint_idx = 0 # next waypoint index
        self.human_marker_pub = rospy.Publisher("human_marker", MarkerArray, queue_size=10)

        rospy.Subscriber('/global_plan', Path, self.global_plan_callback)
        rospy.Subscriber('/waypoint_reached', Bool, self.waypoint_reached_callback)
        rospy.Subscriber('dr_spaam_detections', PoseArray, self.human_callback, queue_size=1) # * static human for now
        rospy.loginfo("Human Checker is initialized.")
        self.need_global_replan = False


    def global_plan_callback(self, msg):
        self.global_path = np.array([
            (single_pose.pose.position.x, single_pose.pose.position.y)
            for single_pose in msg.poses])
        self.path_waypoint_idx = 1
    
    def waypoint_reached_callback(self, msg):
        if self.path_waypoint_idx == len(self.global_path)-1:
            # * goal is reached
            self.global_path = np.array([])
            self.path_waypoint_idx = 0 # next waypoint index
            return
        self.path_waypoint_idx += 1

    def get_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return None, None

        return Point(*trans), rotation[2]

    def human_callback(self, msg):

        try:
            poses_with_headers = [PoseStamped(header=msg.header, pose=pose) for pose in msg.poses]
            # (trans, rot) = self.tf_listener.lookupTransform('map', 'laser', rospy.Time(0)) # ! for real robot
            (trans, rot) = self.tf_listener.lookupTransform('map', 'map', rospy.Time(0)) # ! for gazebo

            transformed_posi = [self.tf_listener.transformPose("/map", pose).pose.position for pose in poses_with_headers]
            if len(self.humans)>0: # original
                human_marker_msg = MarkerArray()
                for human_id in range(len(self.humans)):
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.id = human_id  # Each marker must have a unique ID
                    marker.action = Marker.DELETE
                    human_marker_msg.markers.append(marker)
                self.human_marker_pub.publish(human_marker_msg)

            self.humans = np.array([[pose.x, pose.y] for pose in transformed_posi]) # (n, 2)
            human_marker_msg = MarkerArray()
            for human_id, human in enumerate(self.humans):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.ns = "human"
                marker.id = human_id
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                marker.pose.position.x = human[0]
                marker.pose.position.y = human[1]
                marker.pose.position.z = 0.0  # Z coordinate
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = (0.28+0.01)*2 # 1.0  # Radius of the circle
                marker.scale.y = (0.28+0.01)*2 # 1.0  # Radius of the circle
                marker.scale.z = 0.01
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                human_marker_msg.markers.append(marker)
            self.human_marker_pub.publish(human_marker_msg)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Transform lookup failed.")
        
        if len(self.global_path) == 0:
            # not planned yet
            return

        robot_position, robot_rotation = self.get_pose()
        if robot_position is None:
            rospy.logwarn("Robot pose not available.")
            return
        robot_pos = np.array([robot_position.x, robot_position.y]) # (2,)

        if len(self.humans)==0:
            if self.need_global_replan:
                rospy.wait_for_service("png_navigation/stop_robot")
                stop_robot = rospy.ServiceProxy("png_navigation/stop_robot", StopRobot)
                try:
                    is_stopped = stop_robot()
                    if is_stopped:
                        rospy.loginfo("Robot is stopped upon path collision with human.")
                    else:
                        rospy.loginfo("Robot is not stopped upon path collision with human.")
                except rospy.ServiceException as exc:
                    rospy.loginfo("Service did not process request: " + str(exc))
                rospy.wait_for_service('png_navigation/global_replan')
                try:
                    global_replan_proxy = rospy.ServiceProxy('png_navigation/global_replan', GlobalReplan)
                    replan_human_obstacles = [] # empty
                    response = global_replan_proxy(replan_human_obstacles)
                    if response.is_replanned:
                        rospy.loginfo("Replanned global path.")
                        self.need_global_replan = False
                    else:
                        rospy.loginfo("Failure to replan a global path.")
                        rospy.loginfo(human_obstacles)
                except rospy.ServiceException as e:
                    rospy.logerr("Service call failed: %s" % e)
            return
        humans_to_robot_dist= np.linalg.norm(self.humans-robot_pos, axis=1) # (n,)
        if np.all(humans_to_robot_dist>self.human_detection_radius) \
            and not self.need_global_replan:              
            return
        self.interactive_humans = self.humans[np.where(humans_to_robot_dist<=self.human_detection_radius)] # (m, 2)
        human_obstacles = np.concatenate(
            [self.interactive_humans,
            np.ones((len(self.interactive_humans), 1))*self.human_clearance],
            axis=1,
        ) # (m, 3)
        env_dict = {
            'x_range': [-100,100], # * dummy
            'y_range': [-100,100], # * dummy
            'circle_obstacles': human_obstacles,
            'rectangle_obstacles': [],
        }
        # utils = Utils(Env(env_dict), 0) # * human clearance added to circle radius
        # utils = Utils(Env(env_dict), 0.18) # ! need adjustment on rrt_config human clearance equal robot radius 
        utils = Utils(Env(env_dict), 0.15) # ! need adjustment on rrt_config human clearance equal robot radius 
        if utils.is_inside_obs(robot_pos):
            self.need_global_replan = True
            rospy.loginfo("Robot is within clearance of human obstacles.")
            return

        current_path = np.concatenate([robot_pos[np.newaxis,:], self.global_path[self.path_waypoint_idx:]], axis=0)
        collision = False
        for i in range(len(current_path)-1):
            if np.linalg.norm(current_path[i]-robot_pos)>self.human_detection_radius and \
                np.linalg.norm(current_path[i+1]-robot_pos)>self.human_detection_radius:
                continue
            if utils.is_collision(current_path[i], current_path[i+1]):
                collision = True
                break
        if not collision:
            self.need_global_replan = False
            rospy.wait_for_service("png_navigation/resume_robot")
            resume_robot = rospy.ServiceProxy("png_navigation/resume_robot", ResumeRobot)
            try:
                is_resumed = resume_robot()
                if is_resumed:
                    rospy.loginfo("Robot is resumed upon no path collision with human.")
                else:
                    rospy.loginfo("Robot is not resumed upon no path collision with human.")
            except rospy.ServiceException as exc:
                rospy.loginfo("Service did not process request: " + str(exc))
            return
        self.need_global_replan = True
        rospy.wait_for_service("png_navigation/stop_robot")
        stop_robot = rospy.ServiceProxy("png_navigation/stop_robot", StopRobot)
        try:
            is_stopped = stop_robot()
            if is_stopped:
                rospy.loginfo("Robot is stopped upon path collision with human.")
            else:
                rospy.loginfo("Robot is not stopped upon path collision with human.")
                return
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: " + str(exc))
            return
        rospy.wait_for_service('png_navigation/global_replan')
        try:
            global_replan_proxy = rospy.ServiceProxy('png_navigation/global_replan', GlobalReplan)
            replan_human_obstacles = list(human_obstacles.reshape(-1))
            response = global_replan_proxy(replan_human_obstacles)
            if response.is_replanned:
                rospy.loginfo("Replanned global path.")
                self.need_global_replan = False
            else:
                rospy.loginfo("Failure to replan a global path.")
                rospy.loginfo(human_obstacles)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
        return

if __name__ == '__main__':
    try:
        rospy.init_node('png_navigation_human_checker', anonymous=True)
        hc = HumanChecker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


