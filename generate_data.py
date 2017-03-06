from abc import ABCMeta, abstractmethod
import math

import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns

DEBUG = False 

class Grid():
    """class defining a two-dimensional numpy array
    
    note: conceptually the left-bottom corner of the grid represents the 
          origin (position (0,0)) where the first position represents the
          the x-axis (horizontal) and the second position the y-axis
    """
    __metaclass__ = ABCMeta

    def __init__(self, grid_length, grid_width):
        self.grid_length = grid_length
        self.grid_width = grid_width
        self.grid = np.zeros((grid_length, grid_width))

    def add_noise(self, noise_mean=0.2, noise_std= 0.05):
        self.grid += np.random.normal(noise_mean, noise_std, 
                                      (self.grid_length, self.grid_width))

    def show_grid(self):
        plt.imshow(self.grid, cmap='bone', interpolation='none')
        plt.title(self.type)
        plt.show()

    def check_point_in_grid(self, point):
        return (0 <= point[0] <= self.grid_length-1 and 
                0 <= point[1] <= self.grid_width-1)

    def convert_point_to_be_in_grid(self, point):
        """maps point to closest point in the grid (it can mapped onto itself)

        returns: closest point on the grid, this means that if the point is
                 already in the grid it is returned unchanged
        """
        if self.check_point_in_grid(point):
            return point
        else:
            new_point = [point[0], point[1]]
            if point[0] < 0:
                new_point[0] = 0
            elif point[0] > self.grid_length-1:
                new_point[0] = self.grid_length-1
            if point[1] < 0:
                new_point[1] = 0
            elif point[1] > self.grid_width-1:
                new_point[1] = self.grid_width-1

            return new_point

    def convert_point_to_grid_point(self, point):
        point = self.convert_point_to_be_in_grid(point)
        return [round(point[0]), round(point[1])]

    def get_random_point_in_grid(self, distance_border):
        """ get a random point in the grid that has a min distance to border
            
        returns: a list representing a point in the grid, where the left 
                 upper corner has coordinates 0, 0
        """        
        return [
            np.random.uniform(
                distance_border, self.grid_length - distance_border
            ),
            np.random.uniform(
                distance_border, self.grid_width - distance_border
            )
        ]

    def get_random_point_on_grid(self, distance_border):
        point = self.get_random_point_in_grid(distance_border)
        return [round(point[0]), round(point[1])]

    @abstractmethod
    def fill(self):
        """fill particular region grid according to shape"""
        pass        

class Circle(Grid):
    def __init__(self, grid_width, grid_length, center, radius):
        """the circle object

        params:
            center: an iterable with 2 elements
        """
        super().__init__(grid_width, grid_length)
        if not self.check_point_in_grid(center): 
            raise ValueError("center circle should be within grid")
        self.center = center
        self.radius = radius
        self.type = "circle"
        self.initialize(center, radius)

    def initialize(self, center, radius):
        self.grid = np.zeros(self.grid.shape)  
        left_bottom = self.convert_point_to_be_in_grid(
            [center[0]-radius, center[1]-radius]
        )
        right_top = self.convert_point_to_be_in_grid(
            [center[0]+radius, center[1]+radius]
        )
        if DEBUG:
            print("the left bottom is")
            print(left_bottom)
            print(right_top)
        
        for i in np.arange(int(left_bottom[0]), math.ceil(right_top[0])+1):
            for j in np.arange(int(left_bottom[1]), math.ceil(right_top[1])+1):
                if self.distance([i, j], center) <= radius:
                    self.grid[i, j] = 1  

    def generate_random(self, distance_border, radius_li):        
        radius = np.random.choice(radius_li)
        center = self.get_random_point_on_grid(distance_border)
        self.initialize(center, radius)
        return np.copy(self.grid).flatten()

    @staticmethod
    def distance(point1, point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)  


class Rectangle(Grid):
    def __init__(self, grid_width, grid_length, center, width, length, 
                 rotation):
        """the rectangle object

        params:
            center: iterable with 2 elements first denoting the length 
                    (first axis numpy array) and the second the width
            rotation: should be given in radians
        """
        super().__init__(grid_width, grid_length)
        self.type = "rectangle"
        self.initialize(center, width, length, rotation)
        
    def initialize(self, center, width, length, rotation):
        """fill all points in the grid belonging to the square"""
        self.grid = np.zeros(self.grid.shape)        
        self.rotation_matrix = np.array(
            [[np.cos(-rotation), -np.sin(-rotation)],
            [np.sin(-rotation), np.cos(-rotation)]]
        )
        diag_distance = np.sqrt(width**2 + length**2) 
        up, right, down, left = (
            min(int(center[0]+diag_distance), self.grid_length), 
            min(int(center[1]+diag_distance), self.grid_width),
            max(int(center[0]-diag_distance), 0),
            max(int(center[1]-diag_distance), 0),
        )
        #print("up {}, right {}, down {}, left {}".format(up, right, down, left))
        point = np.zeros(2)
        for i in range(down, up):
            for j in range(left, right):
                point[0], point[1] = i, j
                ro_point = np.absolute(self.rotate(center, point)-center)
                if ro_point[0] <= length/2 and ro_point[1] <= width/2:
                    self.grid[i, j] = 1

    def generate_random(self, distance_border, width_li, length_li,
                        rotation_li):
        center = self.get_random_point_on_grid(distance_border)
        width = np.random.choice(width_li)
        length = np.random.choice(length_li)  
        rotation = np.deg2rad(np.random.choice(rotation_li))
        self.initialize(center, width, length, rotation)        
        return np.copy(self.grid).flatten()

    def rotate(self, center, point):
        return (self.rotation_matrix @ (point-center)) + center 

class Square(Rectangle):
    def __init__(self, grid_width, grid_length, center, size, rotation):
        super().__init__(grid_width, grid_length, center, size, size, rotation)
        self.type = "square"

    def generate_random(self, distance_border, size_li, rotation_li):
        center = self.get_random_point_on_grid(distance_border)
        size = np.random.choice(size_li)        
        rotation = np.deg2rad(np.random.choice(rotation_li))
        self.initialize(center, size, size, rotation) 
        return np.copy(self.grid).flatten()

class GenerateData():    
    def __init__(self, grid_len, grid_width, add_noise=False, params=None):
        self.grid_length = grid_len 
        self.grid_width = grid_width 
        self.add_noise = False 
        self.category_li = np.array(["square", "rectangle", "circle"])
        self.category_dict = {k:count for count, k in enumerate(self.category_li)}
        
        center_dummy = (self.grid_length/2, self.grid_width/2)
        size_dummy = self.grid_length/4
        self.shape_dict = {
            'circle': Circle(grid_width, grid_len, center_dummy, size_dummy),
            'square': Square(grid_width, grid_len, center_dummy, 
                             size_dummy, True),
            'rectangle': Rectangle(grid_width, grid_len, center_dummy, 
                                   size_dummy+1, size_dummy-1, True)
        }

        if params is None:
            size_square_li = [4, 4.5, 5, 5.5, 6, 6.5, 7]
            radius_circle_li = [2.5, 3, 3.5, 4]
            length_rectangle_li = [4.5, 5, 5.5, 6, 6.5, 7]
            width_rectangle_li = [2, 2.5, 3, 3.5, 4]
            rotation_li_rectangle = [0, 45, 90, 135]
            rotation_li_square = [45]
            distance_border = 2

            self.shape_params = {
                'circle': (distance_border, radius_circle_li),
                'square': (distance_border, size_square_li, 
                           rotation_li_square),
                'rectangle': (distance_border, width_rectangle_li, 
                              length_rectangle_li, rotation_li_rectangle)
            }
        
        else:
            self.shape_params = params

    def generate_batch_samples(self, batch_size, shape_prop=None):
        """generate tuple data points plus labels

        returns: a tuple X, y. where X is a numpy array with dimensions 
                 batch_size * (grid_length*grid_width) and y is a numpy 
                 array with dimensions batch_size * (number_of_classes)
        """

        X = np.zeros((batch_size, (self.grid_length*self.grid_width)))
        y = np.zeros((batch_size, len(self.category_li)))
        for i in range(batch_size):
            category = np.random.choice(self.category_li)
            X[i] = self.shape_dict[category].generate_random(
                *self.shape_params[category]
            )
            y[i, self.category_dict[category]] = 1

        return (X, y)

if __name__ == '__main__': 
    #circle = Circle(15, 15, (8, 8), 4)
    #circle.show_grid()

    #rectangle = Rectangle(12, 12, (7.5, 10), 6.5, 3, 0)   
    #rectangle.show_grid()

    #rectangle.initialize((10, 3), 4, 6, (1/4)*np.pi)
    #rectangle.show_grid()

    #square = Square(12, 12, (7, 9), 4, 0)
    #square.show_grid()

    grid_len = 15
    grid_wid = 15
    gen = GenerateData(grid_len, grid_wid)

    samples, y_samples = gen.generate_batch_samples(10)
    for count, sample in enumerate(samples):
        picture = np.reshape(sample, (grid_len, grid_wid))
        plt.imshow(picture, cmap='bone', interpolation='none')
        cat = 0
        for i, entry in enumerate(y_samples[count]):
            if entry ==1 :
                cat = i

        plt.title("the category is {}".format(gen.category_li[cat]))
        plt.show()
    

