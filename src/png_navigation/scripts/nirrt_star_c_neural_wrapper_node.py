#!/home/zhe/miniconda3/envs/pngenv/bin/python
import struct
from os.path import join

import rospy
import rospkg
import numpy as np

from png_navigation.configs.rrt_star_config import Config
from png_navigation.path_planning_classes.rrt_env_2d import Env
from png_navigation.wrapper.pointnet_pointnet2.pointnet2_wrapper_connect_bfs import PNGWrapper as NeuralWrapper
from png_navigation.datasets.point_cloud_mask_utils_updated import generate_rectangle_point_cloud, ellipsoid_point_cloud_sampling

import std_msgs.msg
from sensor_msgs import point_cloud2
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import PointCloud2, PointField

from png_navigation.msg import NIRRTWrapperMsg
from png_navigation.srv import SetEnv, SetEnvResponse


class NeuralWrapperNode:
    def __init__(
        self,
        pc_n_points,
        pc_over_sample_scale,
        clearance,
        step_len,
        connect_max_trial_attempts,
    ):
        rospy.init_node('png_navigation_nirrt_star_c_neural_wrapper_node', anonymous=True)
        package_name = 'png_navigation'
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(package_name)
        root_folderpath = join(package_path, 'src/png_navigation')
        self.neural_wrapper = NeuralWrapper(
            root_dir=root_folderpath,
            device='cuda',
        )
        self.pc_n_points = pc_n_points
        self.pc_over_sample_scale = pc_over_sample_scale
        self.pc_neighbor_radius = step_len
        self.clearance = clearance
        self.connect_max_trial_attempts = connect_max_trial_attempts
        self.pub = rospy.Publisher('wrapper_output', Float64MultiArray, queue_size=10)
        self.guidance_states_pub = rospy.Publisher('guidance_states', PointCloud2, queue_size=10)
        self.no_guidance_states_pub = rospy.Publisher('no_guidance_states', PointCloud2, queue_size=10)
        rospy.Subscriber('wrapper_input', NIRRTWrapperMsg, self.callback, queue_size=1) # * throw away outdated messages
        rospy.Service('png_navigation/neural_wrapper_set_env_2d', SetEnv, self.set_env)

    def set_env(self, request):
        if len(request.request_env.circle_obstacles)>0:
            circle_obstacles = np.array(request.request_env.circle_obstacles).reshape(-1,3)
        else:
            circle_obstacles = []
        if len(request.request_env.rectangle_obstacles)>0:
            rectangle_obstacles = np.array(request.request_env.rectangle_obstacles).reshape(-1,4)
        else:
            rectangle_obstacles = []       
        env_dict = {
            'x_range': request.request_env.x_range,
            'y_range': request.request_env.y_range,
            'circle_obstacles': circle_obstacles,
            'rectangle_obstacles': rectangle_obstacles,
        }
        self.env = Env(env_dict)
        rospy.loginfo("Environment is set for Neural Wrapper.")
        is_set = True
        return SetEnvResponse(is_set)
    
    def callback(self, msg):
        rospy.loginfo("Received request for point net guide inference.")
        x_start = np.array(msg.x_start).astype(np.float64)
        x_goal = np.array(msg.x_goal).astype(np.float64)
        cmax = msg.c_max
        cmin = msg.c_min
        if cmax < np.inf:
            max_min_ratio = cmax/cmin
            pc = ellipsoid_point_cloud_sampling(
                x_start,
                x_goal,
                max_min_ratio,
                self.env,
                n_points=self.pc_n_points,
                n_raw_samples=self.pc_n_points*self.pc_over_sample_scale,
                clearance=self.clearance,
            )
        else:
            pc = generate_rectangle_point_cloud(
                self.env,
                self.pc_n_points,
                over_sample_scale=self.pc_over_sample_scale,
                use_open3d=True,
                clearance=self.clearance,
            )
        fake_env_dict = {}
        fake_env_dict['env_dims'] = [10,10]
        connection_success, _, path_pred = self.neural_wrapper.generate_connected_path_points(
            pc.astype(np.float32),
            x_start,
            x_goal,
            fake_env_dict,
            neighbor_radius=self.pc_neighbor_radius,
            max_trial_attempts=self.connect_max_trial_attempts,
        )
        rospy.loginfo("connection: "+str(connection_success))
        self.path_point_cloud_pred = pc[path_pred.nonzero()[0]] # (<pc_n_points, 2)
        msg = Float64MultiArray()
        msg.data = self.path_point_cloud_pred.flatten()
        self.pub.publish(msg)

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1)
        ]
        points = []
        r = int(1*255)
        g = int(0.4980392156862745*255)
        b = int(0.054901960784313725*255)
        a = 255
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        for i in range(len(self.path_point_cloud_pred)):
            x = self.path_point_cloud_pred[i,0]
            y = self.path_point_cloud_pred[i,1]
            z = 0
            points.append([x, y, z, rgb])
        cloud_msg = point_cloud2.create_cloud(header, fields, points)
        self.guidance_states_pub.publish(cloud_msg)

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1)
        ]
        points = []
        r = 49
        g = 130
        b = 189
        a = 255
        other_point_cloud = pc[(1-path_pred).nonzero()[0]]
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        for i in range(len(other_point_cloud)):
            x = other_point_cloud[i,0]
            y = other_point_cloud[i,1]
            z = 0
            points.append([x, y, z, rgb])
        cloud_msg = point_cloud2.create_cloud(header, fields, points)
        self.no_guidance_states_pub.publish(cloud_msg)


config = Config()
nwn = NeuralWrapperNode(
    config.png_config.pc_n_points,
    config.png_config.pc_over_sample_scale,
    config.png_config.clearance,
    config.png_config.step_len,
    config.png_config.connect_max_trial_attempts,
)
rospy.spin()
      