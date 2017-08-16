def find_optimum_list(opt_list, threshold, tracks, path, path_use=0):
    "find the optimum combination of path and tracks
    
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
    opt_list: a list of lists
        Every list has items: length, height
        
    "
    if threshold < 0:
        raise ValueError("threshold should is smaller zero")
    if path["length"] <= 0 or path["height"] <= 0:
        raise ValueError("the values of path should be higher than zero")
    if threshold == 0:
        return opt_list
    if tracks == []:
        remaining_path_length = min(path["length"]-path_use, threshold)
        opt_list.append(trim_tracks(path, end=remaining_path_length))
        return opt_list

    tracks = trim_tracks(tracks, end=threshold+path_use)
    if path_use != 0 and tracks[0]["height"] > path["height"]:
        raise ValueError()

    while tracks[0]["height"] >= path["height"]:
        threshold -= tracks[0]["length"]
        opt_list.append(tracks.pop(0))

    if threshold == 0:
        return opt_list

    remaining_path_length = min(path["length"]-path_use, threshold)
    length_till_higher_height = find_track_higher_height(tracks, path["height"])
    eq_length = find_eq_length(tracks, path["height"])
    if length_till_higher_height != -1 or length_till_higher_height >= threshold:
        opt_list.append({"length": remaining_path_length, "height": path["height"]})
        opt_list.append(trim_tracks(tracks, end=threshold-remaining_path_length))
        return opt_list
    elif eq_length < path_use and eq_length != -1:
        raise ValueError()
    elif eq_length == path_use:
        use_path = 0
        tracks = trim_tracks(tracks, start=path_use)
        return find_optimum_list(opt_list, threshold, tracks, path, path_use)
    elif eq_length < path["length"] and eq_length != -1:
        add_length = min(remaining_path_length, eq_length-path_use)
        path_use += add_length
        threshold -= add_length
        opt_list.append({"length":add_length, path_height})
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
        length_before_shift, shifted_length = find_shifting_length(tracks, path)
        if threshold < length_before_shift:
            opt_list.append(trim_tracks(tracks, end=threshold)) 
            return opt_list
   
        path_use -= shifted_length
        threshold -= length_before_shift
        opt_list.append(trim_tracks(tracks, end=length_before_shift))
        tracks = trim_tracks(
            tracks, start=length_before_shifted+shifted_length
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
    total_path_height = path["height"]*path["length"]
    if find_height(trim_tracks(tracks, path["length"])) > total_path_height:
        raise ValueError("path length already too high")
    
    higher_track = find_track_higher_height(tracks, path["height"])
    if higher_track == -1:
        return -1, 0

    length_till_higher_track = sum(
        [track["length"] for track in tracks[:higher_track]]
    )
    if path['length'] < length_till_higher_track:
        extra_length += length_till_higher_track - path["length"]
        tracks = trim_tracks(tracks, 
                             start=length_till_higher_track-path["height"])
        return find_shift_length(tracks, path, extra_length)
    elif path['length'] < length_till_higher_track+tracks[higher_track]["length"]:
        next_length = min(
            tracks[0]["length"], 
            length_till_higher_track+tracks[higher_track]["length"]-path["length"]
        )
        old_tracks = trim_tracks[tracks, end=path["length"]]
        new_tracks = trim_tracks(tracks, start=next_length, 
                                 end=path["length"]+next_length)
        total_height_new_tracks = find_height(new_tracks)
        if total_height_new_tracks < total_path_height:
            extra_length += next_length
            trimmed_tracks = trim_tracks(tracks, start=next_length)
            return find_shift_length(trimmed_tracks, path, extra_length)
        elif total_height_new_tracks == total_path_height:
            return next_length+extra_length, path["length"]
        else:
            length_till_shift = (
                (total_path_height-find_height(old_tracks) / 
                (tracks[higher_track]["height"]-tracks[0]["height"])
            )
            return length_till_shift+extra_length, path["length"]
    else:
        tracks_higher_path_length = trim_tracks(tracks, start=path_length)
        next_length = min(tracks[0]["length"], 
                          tracks_higher_path_length[0]["length"])

        last_included_track_old = find_last_included_track(tracks[higher_track+1:], path["height"])
        if tracks_higher_path_length[0]["height"] < path["height"]:
            last_included_track_new = last_included_track_old
        else:
            last_included_track_new = find_last_included_track(tracks[higher_track+1:], path["height"]+next_length)

        use_new_track = last_included_track_old<last_included_track_new
        new_tracks = trim_tracks(tracks[:last_included_track_new+1], start=next_length)
        length_new_tracks = find_length(new_tracks)
        height_new_tracks = find_height(new_tracks)
        if height_new_tracks < length_new_tracks*path["height"]:
            extra_length += next_length
            trimmed_tracks = trim_tracks(tracks, start=next_length)
            return find_shift_length(trimmed_tracks, path, extra_length)
        elif height_new_tracks == length_new_tracks*path["height"]:
            return next_length+extra_length, length_new_tracks
        elif not use_new_track:
            eq_length = (
                (length_new_tracks*path["height"] - height_new_tracks) /
                (path["height"]-tracks[0]["height"])
            )
            return eq_length+extra_length, length_new_tracks
        else:
            trimmed_tracks = trim_tracks(
                tracks,
                start=find_length(tracks[:last_included_track_old]),
                end=path["length"]+next_length
            )
            use_new_track_eq_length = find_eq_length(
                trimmed_tracks, path['height']
            )
            old_tracks = tracks[:last_included_track_old+1]
            first_attempt_eq_length = (
                (path['height']*find_length(old_tracks) - find_height(old_tracks)) / 
                (path_height-tracks[0]["height"])
            )
            if first_attempt_eq_length <= use_new_track_eq_length:
                return (
                    first_attempt_eq_length+extra_length, 
                    find_length(old_tracks)-next_length
                )
            elif:
                second_attempt_eq_length = (
                    (
                        (use_new_track_eq_length*(path_height-tracks[0]["height"]) +
                        (path['height']*find_length(old_tracks) - find_height(old_tracks))
                    ) /
                    (tracks_higher_path_length[0]["height"]- tracks[0]["height"])
                )
                return second_attempt_eq_length+extra_length, path["length"]

def find_last_included_track(tracks, height):
    last_included_track = -1
    total_length, total_height = 0, 0
    for count, track in enumerate(tracks):
        new_total_length = total_length + track["length"]
        new_total_height = total_height + track["height"]
        if new_total_height/new_total_length > height:
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
    for count, track in enumerate(tracks):
        if track["height"] > track:
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
        total_height += track["height"]

    new_total_length = total_length + tracks[higher_track]["length"]
    new_total_height = total_height + tracks[higher_track]["height"]
    if new_total_height/new_total_length > height:
        length_last_track = ((total_length*height - total_height) / 
                             (track["height"] - height))
        return total_length + length_last_track
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

    total_length = 0
    trimmed_tracks = []
    for track in tracks:
        new_total_length = track["length"] + total_length
        if new_total_length < start:
            total_length = new_total_length
        elif new_total_length <= end:
            trimmed_tracks.append([track])
            total_length = new_total_length
        elif total_length <= end and new_total_length >= end:
            trimmed_tracks.append({
                "length":new_total_length-end, 
                "height":track["height"]
            })
            return trimmed_tracks
        else:
            raise ValueError("the loop is not broken at the right moment")

    return trimmed_tracks





