import unittest
import collections
import combining_routes

class TestCombiningRoutes(unittest.TestCase):
    def setUp(self):
        pass

    def test1_find_track_higher_height(self):
        tracks = [
            {"length": 0.7, "height": 1},
            {"length": 1, "height": 2.5},
            {"length": 1.3, "height": 1.8}
        ]
        height = 2
        out = combining_routes.find_track_higher_height(tracks, height)
        self.assertEqual(out, 1)

    def test2_find_track_higher_height(self):
        tracks = [
            {"length": 0.7, "height": 1},
            {"length": 1, "height": 0.5},
            {"length": 1.3, "height": 1.8}
        ]
        height = 2
        out = combining_routes.find_track_higher_height(tracks, height)
        self.assertEqual(out, -1)
       
    def test3_find_track_higher_height(self):
        tracks = [
            {"length": 0.7, "height": 1},
            {"length": 1, "height": 0.5},
            {"length": 1.3, "height": 1.8}
        ]
        height = 1.8
        out = combining_routes.find_track_higher_height(tracks, height)
        self.assertEqual(out, -1)

    def test1_find_height(self):
        tracks = [
            {"length": 0.7, "height": 1},
            {"length": 1, "height": 0.5},
            {"length": 1.3, "height": 2.0}
        ]
        height = combining_routes.find_height(tracks)
        self.assertEqual(height, 3.8)

    def test2_find_height(self):
        height = combining_routes.find_height([])
        self.assertEqual(height, 0)

    def test1_find_length(self):
        tracks = [
            {"length": 2.7, "height": 1},
            {"length": 0.5, "height": 0.5}
        ]
        length = combining_routes.test1_find_length(tracks)
        self.assertEqual(length, 3.2)

    def test1_trim_tracks(self):
        tracks = [
            {"length":0.8, "height": 1},
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.5, "height":2.8}
        ]
        trimmed_tracks = combining_routes.trim_tracks(tracks, start=0.5, end=3.2)
        expected_trimmed_tracks = [
            {"length":0.3, "height": 1},
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.2, "height":2.8}
        ]
        for count, track in enumerate(expected_trimmed_tracks):
            self.assertAlmostEqual(track["length"], trimmed_tracks[count]["length"])
            self.assertAlmostEqual(track["height"], trimmed_tracks[count]["height"])

    def test2_trim_tracks(self):
        tracks = [
            {"length":0.8, "height": 1},
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.5, "height":2.8}
        ]
        trimmed_tracks = combining_routes.trim_tracks(tracks, start=0.8, end=3.2)
        expected_trimmed_tracks = [
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.2, "height":2.8}
        ]
        for count, track in enumerate(expected_trimmed_tracks):
            self.assertAlmostEqual(track["length"], trimmed_tracks[count]["length"])
            self.assertAlmostEqual(track["height"], trimmed_tracks[count]["height"])

    def test3_trim_tracks(self):
        tracks = [
            {"length":0.8, "height": 1},
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.5, "height":2.8}
        ]
        trimmed_tracks = combining_routes.trim_tracks(tracks, start=1.4)
        expected_trimmed_tracks = [
            {"length":0.4 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.5, "height":2.8}
        ]
        for count, track in enumerate(expected_trimmed_tracks):
            self.assertAlmostEqual(track["length"], trimmed_tracks[count]["length"])
            self.assertAlmostEqual(track["height"], trimmed_tracks[count]["height"])

    def test4_trim_tracks(self):
        tracks = [
            {"length":0.8, "height": 1},
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.5, "height":2.8}
        ]
        trimmed_tracks = combining_routes.trim_tracks(tracks, end=2.4)
        expected_trimmed_tracks = [
            {"length":0.8, "height": 1},
            {"length":1 , "height":2.5},
            {"length":0.6, "height":1.8}
        ]
        for count, track in enumerate(expected_trimmed_tracks):
            self.assertAlmostEqual(track["length"], trimmed_tracks[count]["length"])
            self.assertAlmostEqual(track["height"], trimmed_tracks[count]["height"])

     
