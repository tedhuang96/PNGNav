# PNGNav

This is the ROS implementation of NIRRT*-PNG (Neural Informed RRT* with Point-based Network Guidance) for TurtleBot navigation, which is the method in our ICRA 2024 paper

### Neural Informed RRT* with Point-based Network Guidance for Optimal Sampling-based Path Planning

##### [Zhe Huang](https://tedhuang96.github.io/), Hongyu Chen, [Katherine Driggs-Campbell](https://krdc.web.illinois.edu/)

[Paper] [[arXiv](https://arxiv.org/abs/2309.14595)] [Project]

All code was developed and tested on Ubuntu 20.04 with CUDA 12.0, ROS Noetic, conda 23.11.0, Python 3.9.0, and PyTorch 2.0.1. This repo provides the ROS package `png_navigation` which offers rospy implmentations on RRT*, Informed RRT*, Neural RRT*, and our NIRRT*-PNG for TurtleBot navigation. We offer instructions on how to use `png_navigation` in Gazebo simulation, and `png_navigation` can be readily applied in real world scenarios.

The training scripts of PointNet++ and PointNet models for PNG (Point-based Network Guidance) will be released soon.

### Citation
If you find this repo useful, please cite
```
@article{huang2023neural,
  title={Neural Informed RRT* with Point-based Network Guidance for Optimal Sampling-based Path Planning},
  author={Huang, Zhe and Chen, Hongyu and Driggs-Campbell, Katherine},
  journal={arXiv preprint arXiv:2309.14595},
  year={2023}
}
```

## Setup
1. Run
```
cd ~/PNGNav
catkin_make
```

2. Run
```
conda env create -f environment.yml
```

3. Replace `/home/zhe/miniconda3/` in the shebang line of scripts with your own system path. For example, if you are using ubuntu and miniconda3, and your account name is `abc`, replace `/home/zhe/miniconda3/` with `/home/abc/miniconda3`.

4. Download the [PointNet++ model weights](https://drive.google.com/file/d/1YfocGh1pcr_Eg8XhEAxmwaAZQsosjRhM/view?usp=sharing) for PNG, create the folder `model_weights/` in `PNGNav/src/png_navigation/src/png_navigation/wrapper/pointnet_pointnet2/`, and move the downloaded `pointnet2_sem_seg_msg_pathplan.pth` to `model_weights`.
```
cd ~/PNGNav/src/png_navigation/src/png_navigation/wrapper/pointnet_pointnet2/
mkdir model_weights
cd model_weights
mv ~/Downloads/pointnet2_sem_seg_msg_pathplan.pth .
```

5. If you have `map_realworld.pgm` and `map_realworld.yaml`, move them to `PNGNav/src/png_navigation/src/png_navigation/maps`.


## How to Run TurtleBot3 Gazebo Simulation

### Simulation Setup

Follow tutorial on [TurtleBot3 official website](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/) to install dependencies, and test TurtleBot3 gazebo simulation. 

```
sudo apt-get install ros-noetic-joy ros-noetic-teleop-twist-joy \
  ros-noetic-teleop-twist-keyboard ros-noetic-laser-proc \
  ros-noetic-rgbd-launch ros-noetic-rosserial-arduino \
  ros-noetic-rosserial-python ros-noetic-rosserial-client \
  ros-noetic-rosserial-msgs ros-noetic-amcl ros-noetic-map-server \
  ros-noetic-move-base ros-noetic-urdf ros-noetic-xacro \
  ros-noetic-compressed-image-transport ros-noetic-rqt* ros-noetic-rviz \
  ros-noetic-gmapping ros-noetic-navigation ros-noetic-interactive-markers
```

```
sudo apt install ros-noetic-dynamixel-sdk
sudo apt install ros-noetic-turtlebot3-msgs
sudo apt install ros-noetic-turtlebot3
```

```
cd ~/PNGNav/src
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
cd ~/PNGNav && catkin_make
```

### Instructions

0. Add the line below to `~/.bashrc`.
```
export TURTLEBOT3_MODEL=waffle_pi
```

1. Launch Gazebo simulation.
```
cd ~/PNGNav
conda deactivate
source devel/setup.bash
roslaunch turtlebot3_gazebo turtlebot3_world.launch
```

2. Launch `map_server` and `amcl` for Turtlebot3. Note it is the launch file in our `png_navigation` package, which excludes launch of `move_base` and `rviz`.
```
cd ~/PNGNav
conda deactivate
source devel/setup.bash
roslaunch png_navigation turtlebot3_navigation.launch
```

3. Launch rviz.
```
cd ~/PNGNav
conda deactivate
source devel/setup.bash
roslaunch png_navigation rviz_navigation_static.launch
```

4. Estimate the pose of Turtlebot3 by teleoperation. After pose estimation, remember to kill `turtlebot3_teleop_key.launch`.
```
conda deactivate
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

5. Launch the planning algorithm. Start planning on rviz by choosing navigation goal.
```
cd ~/PNGNav
conda deactivate
source devel/setup.bash
roslaunch png_navigation nirrt_star_c.launch
```
or any of these lines
```
roslaunch png_navigation nirrt_star.launch
roslaunch png_navigation nrrt_star.launch
roslaunch png_navigation irrt_star.launch
roslaunch png_navigation rrt_star.launch
```


## How to Run Dynamic Obstacles Implementation

Here are the instructions to run the implementation with dynamic obstacles presented in the simulation. We exchange the terms dynamic obstacles and moving humans.

0. Add the line below to `~/.bashrc`.
```
export TURTLEBOT3_MODEL=waffle_pi
```
1. Launch Gazebo simulation.
```
cd ~/PNGNav
conda deactivate
source devel/setup.bash
roslaunch turtlebot3_gazebo turtlebot3_world.launch
```

2. Launch `map_server` and `amcl` for Turtlebot3. Note it is the launch file in our `png_navigation` package, which excludes launch of `move_base` and `rviz`.
```
cd ~/PNGNav
conda deactivate
source devel/setup.bash
roslaunch png_navigation turtlebot3_navigation.launch
```

3. Launch rviz.
```
cd ~/PNGNav
conda deactivate
source devel/setup.bash
roslaunch png_navigation rviz_navigation_static.launch
```

4. Estimate the pose of Turtlebot3 by teleoperation. After pose estimation, remember to kill `turtlebot3_teleop_key.launch`.
```
conda deactivate
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

5. Create moving humans.
```
cd ~/PNGNav
conda deactivate
source devel/setup.bash
rosrun png_navigation moving_humans_with_noisy_measurements.py
```
Add `/dr_spaam_detections/PoseArray` and `/gt_human_positions/PoseArray` to rviz.

6. Start human detector.
```
cd ~/PNGNav
conda deactivate
source devel/setup.bash
rosrun png_navigation human_checker_gazebo.py
```

7. Start running dynamic obstacle aware planning algorithm.
```
cd ~/PNGNav
conda deactivate
source devel/setup.bash
roslaunch png_navigation nrrt_star_dynamic_obstacles.launch
```
or
```
cd ~/PNGNav
conda deactivate
source devel/setup.bash
roslaunch png_navigation nirrt_star_c_dynamic_obstacles.launch
```

Notes:
- Change moving human speed by changing `vx, vy` in `PNGNav/src/png_navigation/scripts_dynamic_obstacles/moving_humans_with_noisy_measurements.py`.
- change the human detection radius of robot by changing `human_detection_radius` in `PNGNav/src/png_navigation/scripts_dynamic_obstacles/human_checker_gazebo.py`.


## How to Create Your Own Map Yaml File

1. After you finish SLAM and save the map as `.pgm`, you will also get a yaml file. Edit the file which look like this.
```
image: /home/png/map_gazebo.pgm
resolution: 0.010000
origin: [-10.000000, -10.000000, 0.000000]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
setup: 'world'
free_range: [-2, -2, 2, 2]
circle_obstacles: [[1.1, 1.1, 0.15],
                   [1.1, 0, 0.15],
                   [1.1, -1.1, 0.15],
                   [0, 1.1, 0.15],
                   [0, 0, 0.15],
                   [0, -1.1, 0.15],
                   [-1.1, 1.1, 0.15],
                   [-1.1, 0, 0.15],
                   [-1.1, -1.1, 0.15]]
rectangle_obstacles: []
```
The format of fields are as follows.
```
free_range_pixel: [xmin, ymin, xmax, ymax]
circle_obstacles: [[x_center_1, y_center_1, r_1],
                   [x_center_2, y_center_2, r_2],
                   [x_center_3, y_center_3, r_3],
                   ...,
                   [x_center_n, y_center_n, r_n]]
rectangle_obstacles: [[xmin_1, ymin_1, xmax_1, ymax_1],
                      [xmin_2, ymin_2, xmax_2, ymax_2],
                      [xmin_3, ymin_3, xmax_3, ymax_3],
                      ...,
                      [xmin_n, ymin_n, xmax_n, ymax_n]]
```
2. Move `.pgm` and edited `.yaml` files to `PNGNav/src/png_navigation/src/png_navigation/maps`. Keep their names the same, for example `abc.pgm` and `abc.yaml`. When running the launch file, run
```
roslaunch png_navigation nirrt_star_c.launch map:=abc
```
3. You can keep the other fields the same and leave them there. Here is the reference to what these fields mean. If you are going to transform from pixel map to get the geometric configurations, you need these information, and you need `PNGNav/src/png_navigation/src/png_navigation/maps/map_utils.py`.
```
Required fields:

    image : Path to the image file containing the occupancy data; can be absolute, or relative to the location of the YAML file

    resolution : Resolution of the map, meters / pixel

    origin : The 2-D pose of the lower-left pixel in the map, as (x, y, yaw), with yaw as counterclockwise rotation (yaw=0 means no rotation). Many parts of the system currently ignore yaw.

    occupied_thresh : Pixels with occupancy probability greater than this threshold are considered completely occupied.

    free_thresh : Pixels with occupancy probability less than this threshold are considered completely free.

    negate : Whether the white/black free/occupied semantics should be reversed (interpretation of thresholds is unaffected) 
```

## How to Transfer to Your Mobile Robot
1. Modify `robot_config` in `PNGNav/src/png_navigation/src/png_navigation/configs/rrt_star_config.py`. For now, only `robot_config.clearance_radius` matters to global planning.
2. Modify `src/png_navigation/scripts/local_planner_node.py`. In particular, make sure `self.cmd_vel`, `self.odom_frame`, and `self.base_frame` of `class LocalPlanner` match your robot setup. Tune parameters as needed.