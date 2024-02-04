class Env:
    def __init__(
        self,
        env_dict,
        ):
        """
        x_range: tuple (xmin, xmax)
        y_range: tuple (ymin, ymax)
        circle_obstacles: np [[cx, cy, r]] or None
        rectangle_obstacles: np [[x,y,w,h]] or None
        """
        self.x_range = env_dict['x_range']
        self.y_range = env_dict['y_range']
        self.obs_circle = env_dict['circle_obstacles']
        self.obs_rectangle = env_dict['rectangle_obstacles']
        self.dynamic_obs_points = []

    def update_dynamic_obs_points(self, points):
        
        self.dynamic_obs_points = points

    def update_obs_circle_with_ppl_detect(self, points):

        self.obs_circle = points