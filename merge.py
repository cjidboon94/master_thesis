def merge_tracks(tracks1, tracks2):
    """
    Create new tracks which entries are the max of tracks1 and tracks2

    Note: tracks1 and tracks2 should have the same total length. Also
        note that no combination of tracks1 and tracks2 is used.

    Parameters:
    ----------
    tracks1, tracks2: lists of dicts
        Every dict is a track and thus has keys length and height

    Returns: a list of dicts
        Tracks which are the maximum of tracks1 and tracks2

    """
    #lengths have one more item than tracks since they are starting at zero
    lengths1 = create_total_lengths(tracks1)
    lengths2 = create_total_lengths(tracks2)
    if lengths1[-1] != lengths2[-1]:
        raise ValueError("the tracks should have the same total length")
    #print("lengths1 {}, lengths2 {}".format(lengths1, lengths2))
    opt_list = []

    prev_length = 0
    prev_height1, prev_height2 = 0, 0
    new_height1, new_height2 = 0, 0
    pointer1, pointer2 = 0, 0
    new_pointer1, new_pointer2 = 0, 0
    while pointer1 != len(lengths1)-1 or pointer2 != len(lengths2)-1:
        #print("pointer1 {}, pointer2 {}".format(pointer1, pointer2))
        if lengths1[pointer1+1] > lengths2[pointer2+1]:
            extra_length = lengths2[pointer2+1]-prev_length
            new_pointer1 = pointer1
            new_pointer2 = pointer2 + 1
        elif lengths1[pointer1+1] == lengths2[pointer2+1]:
            extra_length = lengths1[pointer1+1]-prev_length
            new_pointer1 = pointer1 + 1
            new_pointer2 = pointer2 + 1
        else:
            extra_length = lengths1[pointer1+1]-prev_length
            new_pointer1 = pointer1 + 1
            new_pointer2 = pointer2

        #print("the extra length {}".format(extra_length))
        #print("new_pointer1 {}, new_pointer2 {}".format(new_pointer1, new_pointer2))

        height_gain1 = extra_length * tracks1[pointer1]["height"]
        height_gain2 = extra_length * tracks2[pointer2]["height"]
        new_height1 = prev_height1 + height_gain1
        new_height2 = prev_height2 + height_gain2

        #print("prev height1: {}, prev height 2: {}".format(prev_height1, prev_height2))
        #print("new height1: {}, new height 2: {}".format(new_height1, new_height2))

        if prev_height1 >= prev_height2 and new_height1 >= new_height2:
            opt_list.append({"length":extra_length, 
                             "height":tracks1[pointer1]["height"]})
        elif prev_height1 <= prev_height2 and new_height1 <= new_height2:
            opt_list.append({"length":extra_length,
                             "height":tracks2[pointer2]["height"]})
        elif prev_height1 > prev_height2 and new_height1 < new_height2:
            prev_height_difference = (prev_height1 - prev_height2)
            gain_diff = tracks2[pointer2]["height"]-tracks1[pointer1]["height"]
            crossing_point = prev_height_difference/gain_diff
            opt_list.append({"length":crossing_point, 
                             "height":tracks1[pointer1]["height"]})
            opt_list.append({"length":extra_length-crossing_point,
                             "height":tracks2[pointer2]["height"]})
        elif prev_height1 < prev_height2 and new_height1 > new_height2:
            prev_height_difference = (prev_height2 - prev_height1)
            gain_diff = tracks1[pointer1]["height"]-tracks2[pointer2]["height"]
            crossing_point = prev_height_difference/gain_diff
            opt_list.append({"length":crossing_point, 
                             "height":tracks2[pointer2]["height"]})
            opt_list.append({"length":extra_length-crossing_point,
                             "height":tracks1[pointer1]["height"]})

        #print("the opt_list: {}".format(opt_list))

        pointer1 = new_pointer1
        pointer2 = new_pointer2
        prev_height1 = new_height1
        prev_height2 = new_height2
        prev_length += extra_length

    opt_list = merge_duplicates(opt_list)
    return opt_list

def merge_duplicates(tracks):
    """merge all consecutive tracks with the same height"""
    if tracks == []:
        return []

    new_tracks = [tracks[0]]
    start_track = 0
    next_track = 1
    while next_track < len(tracks):
        #print("start track {}, next track {}".format(start_track, next_track))
        if tracks[start_track]["height"] == tracks[next_track]["height"]:
            new_tracks[-1] = {
                "length":tracks[start_track]["length"]+tracks[next_track]["length"],
                "height":tracks[start_track]["height"]
            }
            next_track += 1
        else:
            new_tracks.append({
                "length": tracks[next_track]["length"],
                "height": tracks[next_track]["height"]
            })
            start_track, next_track = next_track, next_track+1

    return new_tracks

#obsolete now
def create_length_to_height(tracks, known_lengths, known_heights, all_lengths):
    """
    Create a dict which maps all lengths to known heights

    Note: Zero should always be the starting point!

    Parameters:
    ----------
    tracks: a list of dicts
    known_lengths: a sorted list of numbers
    known_heights: a sorted list of heights
    all_lengths: a sorted list of numbers
        Every entry on known_lengths should also be in all_lengths.
        The last length of all_lengths and known_length should be the same

    """
    length_to_height = {}
    count_known_lengths = 0
    for length in all_lengths:
        #print("length {}".format(length))
        #print("count_known_lengths {}".format(count_known_lengths))
        #zero should always be the starting point
        if length == 0:
            length_to_height[0] = 0.0
            count_known_lengths += 1
        elif length == known_lengths[count_known_lengths]:
            length_to_height[length] = known_heights[count_known_lengths]
            count_known_lengths += 1
        else:
            extra_length = length-known_lengths[count_known_lengths]
            length_to_height[length] = (
                known_heights[count_known_lengths]
                + extra_length*tracks[count_known_lengths-1]["height"]
            )

    return length_to_height

#obsolete now
def merge_lengths(lengths1, lengths2):
    """Merge the sorted lists lengths1 and lengths2 to one sorted list"""
    new_lengths = []
    count1, count2 = 0, 0
    #print("first list length {} second list length {}".format(len(lengths1), len(lengths2)))
    while count1<len(lengths1) or count2<len(lengths2):
        #print("count1 {},  count2 {}".format(count1, count2))
        if count1 == len(lengths1):
            new_lengths.append(lengths2[count2])
            count2 += 1
        elif count2 == len(lengths2):
            new_lengths.append(lengths1[count1])
            count1 +=1
        elif lengths1[count1] < lengths2[count2]:
            new_lengths.append(lengths1[count1])
            count1 += 1
        elif lengths1[count1] == lengths2[count2]:
            new_lengths.append(lengths1[count1])
            count1 += 1
            count2 += 1
        else:
            new_lengths.append(lengths2[count2])
            count2 += 1

    return new_lengths

def create_total_lengths(tracks):
    """
    Create a list that contains for every track in tracks the sum of
    tracks up to the track and up to and including the track
    
    """
    total_length = 0
    total_lengths = [0]
    for track in tracks:
        total_length += track["length"]
        total_lengths.append(total_length)

    return total_lengths

#obsolete now
def create_total_heights(tracks):
    """
    Create a list that contains for every track in tracks the sum of 
    track height up to and containing that track
    
    """
    total_height = 0
    total_heights = [0]
    for track in tracks:
        total_height += track["height"]*track["length"]
        total_heights.append(total_height)

    return total_heights


if __name__ == "__main__":
    tracks1 = [
        {"length":0.2, "height": 1.6},
        {"length":0.32, "height": 1.5},
        {"length":0.98, "height": 2.5},
        {"length":0.3, "height": 2}
    ] 
    tracks2 = [
        {"length":0.2, "height": 1.6},
        {"length":0.4, "height": 2},
        {"length":0.3, "height": 1.5},
        {"length":0.7, "height": 2.5},
        {"length":0.2, "height": 1.2}
    ] 
    
    example_tracks1 = [
        {"length":1, "height": 2},
        {"length":0.5, "height": 1.5},
        {"length":0.2, "height": 1.5},
        {"length":0.8, "height": 1.5},
        {"length":1.6, "height": 3.5},
        {"length": 2, "height": 3},
        {"length": .2, "height": 3},
        {"length": 2, "height": 2.5},
        {"length":0.6, "height": 2.5},
        {"length":0.3, "height": 2},
        {"length":0.3, "height": 1},
        {"length":0.3, "height": 2},
    ]
    #print(merge_duplicates(example_tracks1))

    threshold = 2.4 
    print(merge_tracks(tracks1, tracks2))

    #print(merge_lengths([1, 5, 10, 12, 15, 23], [0, 4, 5, 10, 11, 13, 22, 30]))
    #print(create_length_to_height(
    #    tracks1, [0, 0.2, 0.52, 1.5, 1.8], 
    #    [0, 0.32, 0.8, 3.25, 3.85], [0, 0.2, 0.52, 0.6, 0.9, 1.5, 1.6, 1.8]
    #))
