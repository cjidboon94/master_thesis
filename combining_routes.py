import merge

def find_optimal_height_routes(routes, threshold):
    """return the maximum attained height"""
    opt_list = trim_tracks(routes[0], end=threshold)
    for route in routes:
        opt_list = combine_routes(opt_list, route, threshold)

    return find_height(opt_list)

#still need to be tested
def combine_routes(opt_route, tracks, threshold):
    """create the optimal combination of two routes
    
    Returns: a list with the optimal tracks

    """
    tracks = trim_tracks(tracks, end=threshold)
    new_opt_route = find_optimum_list([], threshold, opt_route, tracks[0])
    remaining_length = threshold-tracks[0]["length"]
    for count, track in enumerate(tracks[1:]):
        temp_opt_route = find_optimum_list(
            tracks[:count+1], remaining_length, 
            trim_tracks(opt_route, end=remaining_length), track
        )
        new_opt_route = merge.merge_tracks(new_opt_route, temp_opt_route)
        remaining_length -= track["length"]

    return new_opt_route


def find_optimum_list(opt_list, threshold, tracks, path, path_use=0):
    """find the optimum combination of path and tracks
    
    Parameters:
    ----------
    tracks: list of dicts
        Every dict has keys length and height
    threshold: a number
        How much length still can added to opt_list.
    path: a dict
        has keys length and height
    path_use: a number
        how much length of the path is already used
    opt_list: a list of dicts
        Every dict has keys length, height. If there are 
        start length and height they should be included as first items
        in optimum list

    Note:
    ----
    start_length + the length of all tracks in tracks should be
    equal (or smaller) than threshold

    Returns:
    -------
    opt_list: a list of dicts
        Every dict has keys: length, height
        
    """
    if threshold < 0:
        raise ValueError("threshold should is smaller zero")
    if path["length"] <= 0 or path["height"] <= 0:
        raise ValueError("the values of path should be higher than zero")
    if threshold == 0:
        return opt_list
    if tracks == []:
        remaining_path_length = min(path["length"]-path_use, threshold)
        opt_list.extend(trim_tracks([path], end=remaining_path_length))
        return opt_list

    tracks = trim_tracks(tracks, end=threshold+path_use)
    if path_use != 0 and tracks[0]["height"] > path["height"]:
        raise ValueError()

    while tracks[0]["height"] >= path["height"]:
        add_length = min(tracks[0]["length"], path_use)
        path_use -= add_length
        threshold -= (tracks[0]["length"]-add_length)
        opt_list.append(tracks.pop(0))
        if tracks == []:
            return find_optimum_list(opt_list, threshold, tracks, path)

    if threshold == 0:
        return opt_list

    remaining_path_length = min(path["length"]-path_use, threshold)
    higher_track = find_track_higher_height(tracks, path["height"])
    if higher_track != -1:
        length_till_higher_height = find_length(tracks[:higher_track])
        eq_length = find_eq_length(tracks, path["height"])
    else:
        length_till_higher_height = -1
        eq_length = -1

    if length_till_higher_height == -1 or length_till_higher_height >= threshold:
        opt_list.append({"length": remaining_path_length, "height": path["height"]})
        opt_list.extend(trim_tracks(tracks, end=threshold-remaining_path_length))
        return opt_list
    elif eq_length < path_use and eq_length != -1:
        raise ValueError()
    elif eq_length == path_use:
        use_path = 0
        tracks = trim_tracks(tracks, start=path_use)
        return find_optimum_list(opt_list, threshold, tracks, path, path_use)
    elif eq_length <= path_use+remaining_path_length and eq_length != -1:
        add_length = min(remaining_path_length, eq_length-path_use)
        threshold -= add_length
        opt_list.append({"length":add_length, "height": path["height"]})
        tracks = trim_tracks(tracks, start=path_use+add_length)
        path_use = 0
        return find_optimum_list(opt_list, threshold, tracks, path, path_use)
    elif (eq_length > path["length"] or eq_length == -1) and remaining_path_length != 0:
        opt_list.append({"length": remaining_path_length, "height": path["height"]})
        threshold -= remaining_path_length
        path_use += remaining_path_length
        return find_optimum_list(opt_list, threshold, tracks, path, path_use)
    elif eq_length > path["length"] or eq_length == -1:
        # a very important note is that either 
        # shifted_length equals path_use and then the first track after
        # adjusting can be higher than path height
        # or
        # the track cannot be higher though it can be equal
        # also length_before_shift must be smaller than length_till_higher_height
        length_before_shift, shifted_length = find_shift_length(tracks, path)
        if threshold < length_before_shift:
            opt_list.extend(trim_tracks(tracks, end=threshold))
            return opt_list
   
        path_use -= shifted_length
        threshold -= length_before_shift
        opt_list.extend(trim_tracks(tracks, end=length_before_shift))
        tracks = trim_tracks(
            tracks, start=length_before_shift+shifted_length
        )
        return find_optimum_list(opt_list, threshold, tracks, path, path_use)

def find_shift_length(tracks, path, extra_length=0):
    """

    Returns: 
    -------
    length_till_shift: a number
        How much length needs to be added before there is some sequence
        with average height equal to the path height. If can't be found
        return -1
    shift_length: a number
        How much length is shifted

    """
    #print("")
    #print("in function the extra lenght is {}".format(extra_length))
    #print(tracks)
    total_path_height = path["height"]*path["length"]
    #print("the total path height is {}".format(total_path_height))
    if find_height(trim_tracks(tracks, end=path["length"])) > total_path_height:
        raise ValueError("path length already too long")
    
    higher_track = find_track_higher_height(tracks, path["height"])
    if higher_track == -1:
        return -1, 0

    old_tracks = trim_tracks(tracks, end=path["length"])
    length_till_higher_track = find_length(tracks[:higher_track])
    #print("length till higher track {}".format(length_till_higher_track))
    if path['length'] < length_till_higher_track:
        #print("in the if")
        extra_length += length_till_higher_track - path["length"]
        tracks = trim_tracks(tracks, 
                             start=length_till_higher_track-path["length"])
        return find_shift_length(tracks, path, extra_length)
    elif path['length'] < length_till_higher_track+tracks[higher_track]["length"]:
        #print("in the elif")
        next_length = min(
            tracks[0]["length"],
            find_length(tracks[:higher_track+1])-path["length"]
        )
        #print("the next length is {}".format(next_length))
        new_tracks = trim_tracks(tracks, start=next_length,
                                 end=path["length"]+next_length)
        height_new_tracks = find_height(new_tracks)
        #print("height new tracks {}".format(height_new_tracks))
        if height_new_tracks < total_path_height:
            extra_length += next_length
            trimmed_tracks = trim_tracks(tracks, start=next_length)
            return find_shift_length(trimmed_tracks, path, extra_length)
        elif height_new_tracks == total_path_height:
            return next_length+extra_length, path["length"]
        else:
            start_difference = total_path_height-find_height(old_tracks)
            gain_per_length = tracks[higher_track]["height"]-tracks[0]["height"]
            #print(start_difference)
            #print(gain_per_length)
            length_till_shift = start_difference/gain_per_length
            return length_till_shift+extra_length, path["length"]
    else:
        #print("in the else")
        tracks_after_path_length = trim_tracks(tracks, start=path["length"])
        next_length = min(
            tracks[0]["length"], tracks_after_path_length[0]["length"]
        )
        #print("the next length is {}".format(next_length))
        new_track = trim_tracks([tracks_after_path_length[0]], end=next_length)[0]

        last_track_old = find_last_relevant_track(old_tracks[higher_track+1:], path["height"])
        #print("the tracks {}".format(old_tracks[higher_track+1:]))
        #print("last tarck old {}".format(last_track_old))
        #accounts for return value of -1, can be used directly in slice notation
        last_included_track_old = higher_track + 1 + last_track_old + 1
        #print("last included track old {}".format(last_included_track_old))
        eq_length_usage_new_track = find_eq_length(
                old_tracks[last_included_track_old:]+[new_track], path["height"]
        )
        if eq_length_usage_new_track == -1:
            optimal_new_tracks = trim_tracks(
                tracks[:last_included_track_old], start=next_length
            )
        else:
            optimal_new_tracks = trim_tracks(old_tracks+[new_track], start=next_length)
            length_previous_not_included_tracks = find_length(old_tracks[last_included_track_old:])
            length_to_add_for_eq = eq_length_usage_new_track-length_previous_not_included_tracks

        #print(eq_length_usage_new_track)
        #print("optimal new tracks")
        #print(optimal_new_tracks)

        optimal_length = find_length(optimal_new_tracks)
        optimal_height = find_height(optimal_new_tracks)
        #print("the optimal  length is {}, height {}".format(optimal_length, optimal_height))
        if optimal_height < optimal_length*path["height"]:
            new_tracks = trim_tracks(tracks, start=next_length)
            return find_shift_length(new_tracks, path, extra_length+next_length)
        elif optimal_height == optimal_length*path["height"]:
            return next_length+extra_length, optimal_length
        else:
            start_difference = (
                path["height"]*find_length(tracks[:last_included_track_old]) -
                find_height(tracks[:last_included_track_old])
            )
            #print("start difference {}".format(start_difference))
            first_gain_per_added_length = path["height"]-tracks[0]["height"]
            #print(first_gain_per_added_length)
            eq_length = start_difference/first_gain_per_added_length
            if eq_length_usage_new_track == -1 or eq_length<length_to_add_for_eq:
                shift_length = optimal_length+(next_length-eq_length)
                return eq_length+extra_length, shift_length
            else:
                start_difference -= (
                    first_gain_per_added_length*length_to_add_for_eq
                )
                #print("start_difference {}".format(start_difference))
                #print(new_track)
                gain_per_added_length = new_track["height"]-tracks[0]["height"]
                #print("gain per added length {}".format(gain_per_added_length))
                new_eq_length = start_difference/gain_per_added_length
                return length_to_add_for_eq+new_eq_length+extra_length, path["length"]

def find_last_relevant_track(tracks, height):
    """
    find the last track so as to optimize:
    sum((track[height]-height)*track[length] for track in tracks)

    Returns: a number

    """
    last_included_track = -1
    total_length, total_height = 0, 0
    for count, track in enumerate(tracks):
        new_total_length = total_length + track["length"]
        new_total_height = total_height + track["height"]*track["length"]
        if new_total_height/new_total_length >= height:
            last_included_track = count
            total_length, total_height = 0, 0
        else:
            total_length = new_total_length
            total_height = new_total_height

    return last_included_track

def find_length(tracks):
    return sum([track["length"] for track in tracks])

def find_height(tracks):
    return sum([track["length"]*track["height"] for track in tracks])

def find_track_higher_height(tracks, height):
    """returns the number of the higher track, counting starts at 0!"""
    for count, track in enumerate(tracks):
        if track["height"] > height:
            return count

    return -1

def find_eq_length(tracks, height, total_length=0, total_height=0):
    """ 
    
    Parameters:
    ----------
    tracks: a list of tracks
    height: a number
    length_already_added: a number
        added for recursion purposes, can be regarded private

    Returns:
    -------
    length: number
        When the weighted average of tracks equals height if not found 
        it has value -1

    """
    higher_track = find_track_higher_height(tracks, height)
    if higher_track == -1:
        return -1
    
    for track in tracks[:higher_track]:
        total_length += track["length"]
        total_height += track["height"]*track["length"]

    new_total_length = total_length + tracks[higher_track]["length"]
    new_total_height = (
        total_height + 
        tracks[higher_track]["height"]*tracks[higher_track]["length"]
    )
    if new_total_height/new_total_length >= height:
        gain_per_extra_length = tracks[higher_track]["height"] - height
        start_difference = total_length*height - total_height
        return start_difference/gain_per_extra_length + total_length
    elif higher_track+1 != len(tracks):
        tracks = tracks[higher_track+1:]
        return find_eq_length(tracks, height, new_total_length, new_total_height)
    else:
        return -1

def trim_tracks(tracks, start=0, end=None):
    """returns the trimmed tracks"""
    end = sum([track["length"] for track in tracks]) if end is None else end
    if start > end:
        raise ValueError("start should not be bigger end")
    if start == end:
        return []

    #print("the start is {}, the end is {}".format(start, end))
    total_length = 0
    trimmed_tracks = []
    for track in tracks:
        new_total_length = track["length"] + total_length
        if new_total_length <= start:
            total_length = new_total_length
        elif total_length < start and new_total_length > start:
            trimmed_tracks.append(
                {
                    "length":new_total_length-start,
                    "height": track["height"]
                }
            )
            total_length = new_total_length
        elif new_total_length < end:
            trimmed_tracks.append(track)
            total_length = new_total_length
        elif new_total_length==end:
            trimmed_tracks.append(track)
            total_length = new_total_length
            return trimmed_tracks
        elif total_length < end and new_total_length >= end:
            trimmed_tracks.append({
                "length":end-total_length, 
                "height":track["height"]
            })
            return trimmed_tracks
        else:
            raise ValueError("the loop is not broken at the right moment")

    return trimmed_tracks

def create_tracks(tracks):
    """create from a list of list a lists of dicts
    
    Parameters:
    ----------
    tracks:a list of pairs
        every pair should be an iterable with two items. The first item 
        equal to the length, the second to the height

    Returns: a list of dicts
        every dict has two keys length and height

    """
    new_tracks = []
    for track in tracks:
        new_tracks.append(
            {
                "length": track[0],
                "height": track[1]
            }
        )
    return new_tracks

if __name__ == "__main__":
    example_tracks = [
        [0.5, 1.5],
        [1, 2.5],
        [0.3, 2.0],
        [0.4, 1.7],
        [0.2, 1.4]
    ]
    tracks = create_tracks(example_tracks)
    path = {"length":0.4, "height":2}
    #print(tracks)
    opt_list = find_optimum_list([{"length":0.2, "height":1.6}], 2.2, tracks, path)
    print(opt_list)
    #trimmed_tracks = trim_tracks(tracks, start=0.5, end=3)
    #print(trimmed_tracks)

