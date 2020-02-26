import unittest
import collections
import merge 

class TestMerge(unittest.TestCase):
    def setUp(self):
        pass

    def test1_merge_tracks(self):
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
        opt_list = merge.merge_tracks(tracks1, tracks2)
        print(opt_list)
        expected_opt_list = [
            {"length":0.2, "height": 1.6},
            {"length":0.4, "height": 2},
            {"length":0.12, "height": 1.5},
            {"length":0.78, "height": 2.5},
            {"length":0.3, "height": 2}
        ] 
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

    def test2_merge_tracks(self):
        tracks1 = [
            {"length":1.5, "height": 3.8},
            {"length":1, "height": 3},
            {"length":0.5, "height": 2.9},
            {"length":0.5, "height": 2.5}
        ] 
        tracks2 = [
            {"length":0.5, "height": 2.9},
            {"length":0.05, "height": 2},
            {"length":1.5, "height": 3.8},
            {"length":1, "height": 3},
            {"length":0.45, "height": 2.7}
        ] 
        opt_list = merge.merge_tracks(tracks1, tracks2)
        print(opt_list)
        expected_opt_list = [
            {"length":1.5, "height": 3.8},
            {"length":1.0, "height": 3},
            {"length":0.5, "height": 2.9},
            {"length":0.175, "height": 2.5},
            {"length":0.325, "height": 2.7}
        ] 
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

