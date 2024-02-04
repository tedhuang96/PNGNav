import math

import numpy as np
import open3d as o3d

from png_navigation.path_planning_classes.collision_check_utils import points_in_circles_rectangles, points_validity

def get_binary_mask(env_img):
    """
    - inputs:
        - env_img: np (img_height, img_width, 3)
        - binary_mask: np float 0. or 1. (img_height, img_width)
    """
    env_dims = env_img.shape[:2]
    binary_mask = np.zeros(env_dims).astype(float)
    binary_mask[env_img[:,:,0]!=0]=1
    return binary_mask


def get_point_cloud_mask_around_points(
    point_cloud,
    points,
    neighbor_radius=3,
    ):
    # point_cloud (n, C)
    # points (m, C), m can be 1
    dist = point_cloud[:,np.newaxis] - points # (n,m,C)
    dist = np.linalg.norm(dist,axis=2) # (n,m) # euclidean distance
    neighbor_mask = dist<neighbor_radius # (n, m)
    around_points_mask = np.sum(neighbor_mask,axis=1)>0 # (n,)
    return around_points_mask

def farthest_point_sample(points, npoint):
    """
    Slower than open3d version.
    Input:
        points: pointcloud data, [B, N, C] or [N, C]
        npoint: number of samples
    Return:
        downsampled_points: [B, npoint, C] or [npoint, C]
        downsampled_indices: sampled pointcloud index, [B, npoint]  or [npoint,]
    """
    points_shape = len(points.shape)
    if points_shape == 2:
        points = points[np.newaxis,:]
    B, N, C = points.shape
    centroids = np.zeros((B, npoint)).astype(int)
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,))
    batch_indices = np.arange(B)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].reshape(B, 1, C)
        dist = np.sum((points - centroid) ** 2, -1) # euclidean distance
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance,-1)
    downsampled_indices = centroids
    downsampled_points_batch_indices = batch_indices[:,np.newaxis]*np.ones((1, npoint)).astype(int)
    downsampled_points = points[downsampled_points_batch_indices, downsampled_indices]
    if points_shape == 2:
        downsampled_points = downsampled_points[0]
        downsampled_indices = downsampled_indices[0]
    return downsampled_points, downsampled_indices

def farthest_point_sample_open3d(points, npoint):
    """
    Input:
        points: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        downsampled_points: [npoint, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.farthest_point_down_sample(num_samples=npoint)
    downsampled_points = np.asarray(pcd.points)
    return downsampled_points


# *** Rectangular sampling ***
def generate_rectangle_point_cloud(
    env,
    n_points,
    over_sample_scale=5,
    use_open3d=True,
    clearance=0,
):
    """
    When creating point cloud, no clearance. Consistent with 2D. We want more points around optimal path to be selected. We may erase them later.
    But we want to keep the topology. If we need to remove topology, we can use clearance, but that is saved for later work.
    - outputs:
        - point_cloud: (n_points, 2)
    """
    point_cloud = np.random.uniform(
        low=(env.x_range[0]+clearance, env.y_range[0]+clearance),
        high=(env.x_range[1]-clearance, env.y_range[1]-clearance),
        size=(n_points*over_sample_scale, 2),
    )
    if len(env.obs_circle)==0:
        obs_circle = None
    else:
        obs_circle = env.obs_circle
    if len(env.obs_rectangle)==0:
        obs_rectangle = None
    else:
        obs_rectangle = env.obs_rectangle
    in_obs = points_in_circles_rectangles(
        point_cloud,
        obs_circle,
        obs_rectangle,
        clearance=clearance, # * can be adjusted
    )
    point_cloud = point_cloud[(1-in_obs).astype(bool)]
    if len(point_cloud) > n_points:
        if use_open3d:
            point_cloud_fake_z = np.concatenate([point_cloud, np.zeros((point_cloud.shape[0],1))],axis=1) # (n,3)
            point_cloud_fake_z = farthest_point_sample_open3d(point_cloud_fake_z, n_points)
            point_cloud = point_cloud_fake_z[:,:2]
        else:
            point_cloud, _ = farthest_point_sample(point_cloud, n_points)
    return point_cloud


# *** Ellipse sampling ***
def RotationToWorldFrame(start_point, goal_point, L):
    """
    - inputs:
        - start_point: np float64 (2,)
        - goal_point: np float64 (2,)
        - L: scalar
    - outputs:
        - C: rotation matrix, np float64 (3,3)
    """
    a1 = (goal_point - start_point)/L
    a1 = np.concatenate([a1, np.array([0.])], axis=0)[:,np.newaxis] # (3,1)
    e1 = np.array([[1.0], [0.0], [0.0]])
    M = a1 @ e1.T
    U, _, V_T = np.linalg.svd(M, True, True)
    C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T
    return C

def get_distance_and_angle(start_point, goal_point):
    """
    - inputs:
        - start_point: np float64 (2,)
        - goal_point: np float64 (2,)
    """
    dx, dy = goal_point - start_point
    return math.hypot(dx, dy), math.atan2(dy, dx)


def ellipsoid_point_cloud_sampling(
    start_point,
    goal_point,
    max_min_ratio,
    env,
    n_points=1000,
    n_raw_samples=10000,
    clearance=0,
):
    """
    - inputs
        - start_point: np (2,)
        - goal_point: np (2,)
        - max_min_ratio: scalar >= 1.0
        - env
    - outputs
        - point_cloud: np (n_points, 2)
    """
    c_min, theta = get_distance_and_angle(start_point, goal_point)
    C = RotationToWorldFrame(start_point, goal_point, c_min)
    x_center = (start_point+goal_point)/2.
    x_center = np.concatenate([x_center, np.array([0.])], axis=0) # (3,)
    c_max = c_min*max_min_ratio
    if c_max ** 2 - c_min ** 2<0:
        eps = 1e-6
    else:
        eps = 0
    r = [c_max / 2.0,
        math.sqrt(c_max ** 2 - c_min ** 2+eps) / 2.0,
        math.sqrt(c_max ** 2 - c_min ** 2+eps) / 2.0]
    L = np.diag(r)

    samples = np.random.uniform(-1, 1, size=(n_raw_samples, 2))
    samples = samples[np.linalg.norm(samples, axis=1) <= 1]
    samples = np.concatenate([samples, np.zeros((len(samples),1))], axis=1) # (n, 3)

    x_rand = np.dot(np.dot(C, L), samples.T).T + x_center
    point_cloud = x_rand[:,:2]

    if len(env.obs_circle)>0:
        obs_circle = np.array(env.obs_circle).astype(np.float64)
    else:
        obs_circle = None
    if len(env.obs_rectangle)>0:
        obs_rectangle = np.array(env.obs_rectangle).astype(np.float64)
    else:
        obs_rectangle = None
    valid_flag = points_validity(
        point_cloud,
        obs_circle,
        obs_rectangle,
        env.x_range,
        env.y_range,
        obstacle_clearance=clearance,
        range_clearance=clearance,
    )
    point_cloud = point_cloud[valid_flag]

    if len(point_cloud) > n_points:
        # downsample
        point_cloud_fake_z = np.concatenate([point_cloud, np.zeros((point_cloud.shape[0],1))],axis=1) # (n,3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_fake_z)
        pcd = pcd.farthest_point_down_sample(num_samples=n_points)
        point_cloud = np.asarray(pcd.points)[:,:2]
    return point_cloud



