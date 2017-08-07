import numpy as np
import copy
from Collections import OrderedDict

class Path():
    def __init__(self, length, height, route_number, start_track, end_track, path):
        self.path = path
        self.height_per_length = height/length
        self.length = length
        self.height = height
        self.route_number = route_number
        self.start_track = start_track
        self.end_track = end_track

    def get_height(self, threshold):
        total_length, total_height = 0, 0
        for length, height in self.path:
            if length+total_length<threshold:
                total_length += length
                total_height += height
            else:
                total_height += height * ((length+total_length-threshold)/length)
                break

        return total_height

class Route():
    def __init__(self, route, threshold, route_number):
        """
        A route consists of a number of consecutive tracks with a
        length and a height. No track can be taken if not first all
        tracks before that track have been taken. Tracks can be partly
        taken. 

        Parameters:
        ----------
        route: a list
            Every route consist of a list of tuples representing 
            tracks. Every track has a length, the first item, and a 
            height. If a track has a length longer than the threshold
            it will rounded down to threshold.
        threshold: a number

        """
        self.threshold = threshold
        self.route_number = route_number
        self.route = []
        count = 0
        total_length = 0
        for length, height in route:
            if total_length+length =< threshold:
                self.route.append([length, height])
                total_length += length
            else:
                self.route.append([threshold-total_length, height])
                break
    
    def find_path(self, start):
        count, total_length, total_height = start, 0, 0
        max_height_per_length, maximum_amount_of_tracks = 0, start
        for length, height in route[start:]:
            total_length += length
            total_height += height
            height_per_length = total_height/total_length
            if height_per_length > max_height_per_length:
                max_length = total_length
                max_height = total_height
                maximum_amount_of_tracks = count

            count += 1
        
        return Path(max_length, max_height, self.route_number, start, 
                    maximum_amount_of_tracks, route[start:maximum_amount_of_tracks+1])

    def create_paths(self):
        self.paths = []
        start, end = 0, len(self.route)
        while start < end:
            path, last_track = self.find_path(start)
            self.paths.append(path)
            start = last_track + 1
 
def find_height_combination(route, combination, paths):
    pass

def find_all_combinations_under_threshold():
    combinations = []
    for route_number, paths in route_to_chosen_paths.items():
        if paths[-1].length => overlength:
            combinations.append([route_number, len(paths)-1])
            continue
        else:
            pass

def find_best_subset(routes, route_to_length, threshold, paths):
    overlength = sum([route_to_length[i] for i in range(len(routes))])-treshold
    route_to_chosen_paths = find_paths_on_route(paths)

    combinations = find_all_combinations_under_threshold():
    heights = [[combination, find_height_combination(combination)] for combination in combinations]
    best_combination = min(heights, key=lambda x: x[0])
    #still need to adjust for routes
    return best_combination


def find_paths_on_route(paths):
    route_to_chosen_paths = OrderedDict()
    for path in paths:
        if path.route_number in route_to_chosen_paths:
            route_to_chosen_paths[path.route_number].append(path)
        else:
            route_to_chosen_paths[path.route_number] = [path]
    
    return route_to_chosen_paths

def find_route_length(routes, paths):
    route_to_length = {}
    for path in paths:
        if path.route_number in route_to_length:
            route_to_length[path.route_number] += path.length
        else:
            route_to_lengths[path.route_number] = path.length

    return route_to_length

def find_best_paths(tracks_list, threshold):
    """
    Find the combination of tracks in the routes that have a combined
    length equal to threshold with maximal height.

    Parameters:
    ----------
    routes: a list of Route objects
        The combined length of all route objects should be bigger than
        the threshold. Otherwise a ValueError is thrown
    threshold: a number

    Returns: min_max_impact, max_max_impact
    -------
    lower_bound_max_impact: a number
    upper_bound_max_impact: a number

    """
    routes = []
    for tracks in tracks_list:
        route = Route(tracks)
        route.create_paths()
        routes.append(route)

    paths = [route.paths for route in routes]
    paths_sorted = sorted(paths, key=lambda x: x.height_per_lenght)
    count, length = 0, 0
    while length < threshold:
        length += paths_sorted[count].length
        count += 1

    route_to_length = find_route_lengths(routes, paths)
    overlength = sum([route_to_length[i] for i in range(len(routes))])-treshold
    route_to_paths = find_paths_on_route(paths)

    upper_bound_max_impact = (sum([path.height for path in paths[:-1]]) +
                      paths[-1].height*((paths[-1].length-overlength)/path.length)*)
    lower_bound_max_impact = (sum([path.height*path.length for path in paths[:-1]]) +
                      paths[-1].get_height(paths[-1].length-overlength))

    return lower_bound_max_impact, upper_bound_max_impact

    #total_height, route_length = find_best_subset(routes, route_to_length, threshold, paths)

