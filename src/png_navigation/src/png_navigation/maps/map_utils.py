import math

import numpy as np

from png_navigation.path_planning_classes.collision_check_utils import points_in_circles_rectangles


def approximate_free_vol_2d(env, clearance=0, n_points=100000):
    points = np.array((
        np.random.uniform(env.x_range[0]+clearance, env.x_range[1]-clearance, n_points),
        np.random.uniform(env.y_range[0]+clearance, env.y_range[1]-clearance, n_points),
    )).T # (n_points, 2)

    in_obs = points_in_circles_rectangles(
        points,
        env.obs_circle,
        env.obs_rectangle,
        clearance=clearance,
    ).astype(np.float64) # (n_points, )
    free_vol_approx_ratio = 1-np.mean(in_obs)
    free_vol = (env.x_range[1]-env.x_range[0]-2*clearance)*(env.y_range[1]-env.y_range[0]-2*clearance)*free_vol_approx_ratio
    return free_vol

def compute_gamma_rrt_star_2d(env, clearance=0):
    dim = 2
    unit_ball_vol = np.pi # pi*1**2
    free_vol = approximate_free_vol_2d(env, clearance)
    return math.ceil((2*(1+1./dim))**(1./dim)*(free_vol/unit_ball_vol)**(1./dim))


def get_transform_pixel_to_world(
    map_config,
    map_image,
):
    # [xw, yw].T = A_wp @ [xp, yp].T + b_wp.T # w -> world, p -> pixel
    A_wp = np.array([[map_config['resolution'], 0], [0, -map_config['resolution']]]) # (2,2)
    xw, yw = map_config['origin'][:2] # origin: The 2-D pose of the lower-left pixel in the map
    img_width, img_height = map_image.size
    xp, yp = 0, img_height-1
    b_wp = (np.array([[xw],[yw]]) - A_wp.dot(np.array([[xp],[yp]])))[:,0] # (2,)
    xp_origin, yp_origin = int(b_wp[0]/(-map_config['resolution'])), int(b_wp[1]/map_config['resolution'])
    return A_wp, b_wp, (xp_origin, yp_origin)

def pixel_to_world_coordinates(points_p, A_wp, b_wp):
    """
    - inputs:
        - points_p: (n, 2)
        - A_wp: (2, 2)
        - b_wp: (2,)
    - outputs:
        - points_w: (n, 2)
    """
    return (A_wp.dot(points_p.T)).T+b_wp

def min_max_aabb(aabb):
    """
    aabb: (4,) or (n,4) for (x1,y1,x2,y2)
    """
    if len(aabb.shape)==1:
        return np.array([min(aabb[0],aabb[2]), min([aabb[1],aabb[3]]), max(aabb[0],aabb[2]), max([aabb[1],aabb[3]])])
    xmin = np.min(np.stack([aabb[:,0], aabb[:,2]], axis=1),axis=1) # (n,)
    xmax = np.max(np.stack([aabb[:,0], aabb[:,2]], axis=1),axis=1) # (n,)
    ymin = np.min(np.stack([aabb[:,1], aabb[:,3]], axis=1),axis=1) # (n,)
    ymax = np.max(np.stack([aabb[:,1], aabb[:,3]], axis=1),axis=1) # (n,)
    return np.stack([xmin,ymin,xmax,ymax],axis=1) # (n,4)
