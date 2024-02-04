#!/usr/bin/python3.8
import rospy
from std_msgs.msg import String

rospy.init_node('png_navigation_local_planner_clock', anonymous=True)
desired_frequency = 20
rate = rospy.Rate(desired_frequency)
pub = rospy.Publisher('png_navigation/local_planner_clock', String, queue_size=10)
while not rospy.is_shutdown():
    message = "dummy msg for local planner"
    pub.publish(message)
    rate.sleep()