#!/usr/bin/python3.8
import math

import tf
import rospy
import numpy as np
from tf.transformations import euler_from_quaternion

from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist, Point, PoseStamped


def normalize_angle(angle):
    normalized_angle = math.fmod(angle + math.pi, 2 * math.pi)
    if normalized_angle < 0:
        normalized_angle += 2*math.pi
    return normalized_angle - math.pi

class LocalPlanner:
    def __init__(
        self,
        robot_frame='base_footprint',
        max_linear_speed=0.2,
        max_angular_speed=(0.2, 0.4),
        distance_threshold=(0.05, 0.3),
        angle_threshold=(0.05, 0.1),
    ):
        rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5) # gazebo
        # * self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=5) # real world
        self.waypoint_reached_pub = rospy.Publisher('/waypoint_reached', Bool, queue_size=1)

        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'map'
        self.base_frame = robot_frame

        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0
        self.is_global_goal = False

        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
       
        self.linear_speed_increment = 0.01#0.005
        self.angular_speed_increment = 0.1
        self.linear_speed = 0
        self.angular_speed = 0

        self.target_linear_speed = 0
        self.target_angular_speed = 0

        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.drive_robot = False

        rospy.Subscriber('/waypoint', PoseStamped, self.waypoint_callback)
        rospy.Subscriber('png_navigation/local_planner_clock', String, self.clock_callback)
        rospy.loginfo("Local Planner is initialized.")

    def waypoint_callback(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        if msg.pose.position.z != 0:
            self.is_global_goal = True
            angles = tf.transformations.euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            self.goal_yaw = angles[-1]
        self.drive_robot = True
    
    def clock_callback(self, msg):
        if not self.drive_robot:
            return
        position, rotation = self.get_pose()
        distance = np.sqrt((self.goal_x - position.x)**2 + (self.goal_y - position.y)**2)
     
        if distance < self.distance_threshold[0]:
            if not self.is_global_goal:
                rospy.loginfo("Waypoint reached.")
                self.waypoint_reached_pub.publish(True)
                return
            else:
                self.target_linear_speed = 0
                remaining_rotation = normalize_angle(self.goal_yaw - rotation)
                if abs(remaining_rotation) > self.angle_threshold[1]:
                    if remaining_rotation > 0:
                        self.target_angular_speed = 0.4
                    else:
                        self.target_angular_speed = -0.4
                    self.send_velocity_command(self.target_linear_speed, self.target_angular_speed)
                    return
                elif abs(remaining_rotation) > self.angle_threshold[0]:
                    if remaining_rotation > 0:
                        self.target_angular_speed = 0.2
                    else:
                        self.target_angular_speed = -0.2
                    self.send_velocity_command(self.target_linear_speed, self.target_angular_speed)
                    return
                else:
                    self.target_angular_speed = 0
                    self.send_velocity_command(self.target_linear_speed, self.target_angular_speed)
                    if self.linear_speed==0 and self.angular_speed==0:
                        rospy.loginfo("Global goal reached announced by local planner.") # global goal reached
                        self.waypoint_reached_pub.publish(True)
                        self.drive_robot = False
                        self.goal_yaw = None
                        self.is_global_goal = False
                        return
        
        path_angle = np.arctan2(self.goal_y - position.y, self.goal_x - position.x)
        remaining_rotation = normalize_angle(path_angle - rotation)
        if abs(remaining_rotation) > self.angle_threshold[1]:
            self.target_linear_speed = 0.
            if remaining_rotation > 0:
                self.target_angular_speed = 0.4
            else:
                self.target_angular_speed = -0.4
            self.send_velocity_command(self.target_linear_speed, self.target_angular_speed)
            return
        elif abs(remaining_rotation) > self.angle_threshold[0]:
            if remaining_rotation > 0:
                self.target_angular_speed = 0.2
            else:
                self.target_angular_speed = -0.2

            if distance < self.distance_threshold[1] and self.is_global_goal:
                self.target_linear_speed = 0.05
            else:
                self.target_linear_speed = 0.2
            self.send_velocity_command(self.target_linear_speed, self.target_angular_speed)
            return
        else:
            self.target_angular_speed = 0.
            if distance < self.distance_threshold[1] and self.is_global_goal:
                self.target_linear_speed = 0.1
            else:
                self.target_linear_speed = 0.2
            self.send_velocity_command(self.target_linear_speed, self.target_angular_speed)
            return
            
    def send_velocity_command(self, target_linear_speed, target_angular_speed):
        if abs(target_linear_speed-self.linear_speed) < self.linear_speed_increment:
            self.linear_speed = target_linear_speed
        else:
            self.linear_speed += self.linear_speed_increment*(target_linear_speed-self.linear_speed)/abs(self.target_linear_speed-self.linear_speed)
        if abs(target_angular_speed-self.angular_speed) < self.angular_speed_increment:
            self.angular_speed = target_angular_speed
        else:
            self.angular_speed += self.angular_speed_increment*(target_angular_speed-self.angular_speed)/abs(self.target_angular_speed-self.angular_speed)
        move_cmd = Twist()
        move_cmd.linear.x = self.linear_speed
        move_cmd.angular.z = self.angular_speed
        self.cmd_vel.publish(move_cmd)

    def get_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return
        return Point(*trans), rotation[2]

    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)


if __name__ == '__main__':
    try:
        rospy.init_node('png_navigation_local_planner', anonymous=True)
        gp = LocalPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


