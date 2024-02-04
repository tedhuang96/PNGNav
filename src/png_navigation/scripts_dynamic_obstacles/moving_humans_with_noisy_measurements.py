#!/usr/bin/python3.8
import time

import rospy
import numpy as np

from geometry_msgs.msg import Pose, PoseArray


def publish_pose_array():
    rospy.init_node('pose_array_publisher')
    pose_array_pub = rospy.Publisher('dr_spaam_detections', PoseArray, queue_size=10)
    real_pose_array_pub = rospy.Publisher('gt_human_positions', PoseArray, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz
    px, py = 1, -0.8
    count_turn = 40
    vx, vy = -1/20., 0
    count = 0
    ts = time.time()

    while not rospy.is_shutdown():
        pose_array_msg = PoseArray()
        real_pose_array_msg = PoseArray()
        if count > count_turn:
            vx = -vx
            count = 0
        px += vx
        py += vy
        count += 1

        for i in range(1):
            pose = Pose()
            pose.position.x = px+np.random.randn()*0.01
            pose.position.y = py+np.random.randn()*0.01
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
            pose_array_msg.poses.append(pose)

        if time.time() - ts > 2:
            pose = Pose()
            pose.position.x = -1.5+np.random.randn()*0.01
            pose.position.y = 0.5+np.random.randn()*0.01
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
            pose_array_msg.poses.append(pose)   

        pose_array_msg.header.stamp = rospy.Time.now()
        pose_array_msg.header.frame_id = "map"
        pose = Pose()
        pose.position.x = px
        pose.position.y = py
        pose.position.z = 0.0
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        real_pose_array_msg.poses.append(pose)

        if time.time() - ts > 2:
            pose = Pose()
            pose.position.x = -1.5
            pose.position.y = 0.5
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
            real_pose_array_msg.poses.append(pose)   
        real_pose_array_msg.header.stamp = rospy.Time.now()
        real_pose_array_msg.header.frame_id = "map"
        pose_array_pub.publish(pose_array_msg)
        real_pose_array_pub.publish(real_pose_array_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_pose_array()
    except rospy.ROSInterruptException:
        pass