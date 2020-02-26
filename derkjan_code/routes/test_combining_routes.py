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
        length = combining_routes.find_length(tracks)
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

    def test5_trim_tracks(self):
        tracks = [
            {'length': 0.0011092335708519701, 'height': 0.6977046642789908}, 
            {'length': 0.0088907664291480309, 'height': 0.88774757794532244}
        ]
        new_tracks = combining_routes.trim_tracks(tracks, end=0.01)
        threshold = 0.01
        total_length = new_tracks[0]["length"]+new_tracks[1]["length"]
        self.assertEqual(total_length-(total_length-threshold), threshold)

    def test1_find_eq_length(self):
        tracks = [
            {"length":0.8, "height": 1},
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.5, "height":2.8}
        ]
        eq_length = combining_routes.find_eq_length(tracks, 3)
        self.assertEqual(eq_length, -1)

    def test2_find_eq_length(self):
        tracks = [
            {"length":0.8, "height": 1},
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.5, "height":2.6}
        ]
        eq_length = combining_routes.find_eq_length(tracks, 2.4)
        self.assertEqual(eq_length, -1)

    def test3_find_eq_length(self):
        tracks = [
            {"length":0.8, "height": 1},
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.5, "height":2.8}
        ]
        eq_length = combining_routes.find_eq_length(tracks, 0.5)
        self.assertEqual(eq_length, 0)
    
    def test4_find_eq_length(self):
        tracks = [
            {"length":0.8, "height": 1},
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.5, "height":2.8}
        ]
        eq_length = combining_routes.find_eq_length(tracks, 1.5)
        self.assertAlmostEqual(eq_length, 1.2)
    
    def test5_find_eq_length(self):
        tracks = [
            {"length":0.8, "height": 1},
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.9, "height":2.8}
        ]
        eq_length = combining_routes.find_eq_length(tracks, 2)
        self.assertAlmostEqual(eq_length, 3.675)

    def test6_find_eq_length(self):
        tracks = [
            {"length":0.8, "height": 1},
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.675, "height":2.8}
        ]
        eq_length = combining_routes.find_eq_length(tracks, 2)
        self.assertAlmostEqual(eq_length, 3.675)

    def test7_find_eq_length(self):
        tracks = [
            {"length":0.8, "height": 1},
            {"length":1 , "height":2.5},
            {"length":1.2, "height":1.8},
            {"length":0.67, "height":2.8}
        ]
        eq_length = combining_routes.find_eq_length(tracks, 2)
        self.assertAlmostEqual(eq_length, -1)

    def test8_find_eq_length(self):
        tracks = []
        eq_length = combining_routes.find_eq_length(tracks, 2)
        self.assertAlmostEqual(eq_length, -1)

    def test1_find_last_relevant_track(self):
        tracks = [
            {"length":0.5, "height": 1.4},
            {"length":0.8 , "height":2},
            {"length":0.2, "height":3.5},
            {"length":0.6, "height":2.4}
        ]
        last_included_track = combining_routes.find_last_relevant_track(tracks, 4)
        self.assertEqual(last_included_track, 0)

    def test2_find_last_relevant_track(self):
        tracks = [
            {"length":0.5, "height": 1.4},
            {"length":0.8 , "height":2},
            {"length":0.2, "height":3.5},
            {"length":0.6, "height":2.4}
        ]
        last_included_track = combining_routes.find_last_relevant_track(tracks, 3)
        self.assertEqual(last_included_track, 0)

    def test3_find_last_relevant_track(self):
        tracks = [
            {"length":0.5, "height": 1.4},
            {"length":0.8 , "height":2},
            {"length":0.2, "height":3.5},
            {"length":0.6, "height":2.4}
        ]
        last_included_track = combining_routes.find_last_relevant_track(tracks, 1.8)
        self.assertEqual(last_included_track, 4)

    def test4_find_last_relevant_track(self):
        tracks = [
            {"length":0.5, "height": 1.4},
            {"length":0.8 , "height":2},
            {"length":0.8, "height":4},
            {"length":0.6, "height":2.4}
        ]
        last_included_track = combining_routes.find_last_relevant_track(tracks, 2.5)
        self.assertEqual(last_included_track, 3)
   
    def test1_find_shift_length(self):
        tracks = [
            {"length":0.5, "height": 1.4},
            {"length":0.8 , "height":2},
            {"length":0.8, "height":3},
            {"length":0.6, "height":2.4}
        ]
        path = {"length":2 , "height":3.5}
        length_till_shift, shift_length = combining_routes.find_shift_length(tracks, path)
        self.assertEqual(length_till_shift, -1)
        self.assertEqual(shift_length, 0)

    def test2_find_shift_length(self):
        tracks = [
            {"length":0.5, "height": 1.4},
            {"length":0.8 , "height":2},
            {"length":0.8, "height":3},
            {"length":0.6, "height":2.4}
        ]
        path = {"length":2 , "height":0.8}
        self.assertRaises(Exception, combining_routes.find_shift_length, tracks, path)

    def test3_find_shift_length(self):
        tracks = [
            {"length":0.5, "height": 1.4},
            {"length":0.8 , "height":2},
            {"length":0.8, "height":3},
            {"length":0.6, "height":2.4}
        ]
        path = {"length":0.9 , "height":1.7}
        length_till_shift, shift_length = combining_routes.find_shift_length(tracks, path)
        self.assertAlmostEqual(length_till_shift, 0.05)
        self.assertAlmostEqual(shift_length, 0.9)

    def test4_find_shift_length(self):
        tracks = [
            {"length":1.0, "height": 1.4},
            {"length":0.8 , "height":2},
            {"length":0.8, "height":3},
            {"length":0.6, "height":2.4}
        ]
        path = {"length":0.9 , "height":1.7}
        length_till_shift, shift_length = combining_routes.find_shift_length(tracks, path)
        self.assertAlmostEqual(length_till_shift, 0.55)
        self.assertAlmostEqual(shift_length, 0.9)

    def test5_find_shift_length(self):
        tracks = [
            {"length":1.0, "height": 1},
            {"length":0.7 , "height":2.5},
            {"length":0.8, "height":1.8},
            {"length":0.6, "height":2.4}
        ]
        path = {"length":0.9 , "height":2.0}
        length_till_shift, shift_length = combining_routes.find_shift_length(tracks, path)
        print(length_till_shift, shift_length)
        self.assertAlmostEqual(length_till_shift, 0.7)
        self.assertAlmostEqual(shift_length, 0.9)

    def test6_find_shift_length(self):
        tracks = [
            {"length":0.3, "height": 0.5},
            {"length":0.5, "height": 0.4},
            {"length":1.5, "height": 2.5},
            {"length":0.6, "height": 2.4}
        ]
        path = {"length":1.0, "height":1.8}
        length_till_shift, shift_length = combining_routes.find_shift_length(tracks, path)
        print(length_till_shift, shift_length)
        self.assertAlmostEqual(length_till_shift, 0.466666666)
        self.assertAlmostEqual(shift_length, 1.0)

    def test7_find_shift_length(self):
        tracks = [
            {"length":0.7, "height": 1},
            {"length":0.6, "height": 1.5},
            {"length":0.5, "height": 2.5},
            {"length":1, "height": 1.7}
        ]
        path = {"length":1.4, "height":1.8}
        length_till_shift, shift_length = combining_routes.find_shift_length(
            tracks, path
        )
        print(length_till_shift, shift_length)
        self.assertAlmostEqual(length_till_shift, 0.4875)
        self.assertAlmostEqual(shift_length, 1.8-0.4875)

    def test8_find_shift_length(self):
        tracks = [
            {"length":1.0, "height": 2},
            {"length":0.8, "height": 1.6},
            {"length":0.9, "height": 3.0},
            {"length":0.4, "height": 2.2},
            {"length":0.6, "height": 3.2}
        ]
        path = {"length":3, "height":2.4}
        length_till_shift, shift_length = combining_routes.find_shift_length(tracks, path)
        print(length_till_shift, shift_length)
        self.assertAlmostEqual(length_till_shift, 0.55)
        self.assertAlmostEqual(shift_length, 3)

    def test9_find_shift_length(self):
        tracks = [
            {'length': 0.001357555252636717, 'height': 0.27323144259972437},
            {'length': 0.00044454299901235724, 'height': 0.29143638564699315},
            {'length': 0.0015111441942070921, 'height': 0.28761205180112825},
            {'length': 0.0044636256720058352, 'height': 0.29966298108692219}
        ]
        path = {'length': 0.0022231318821379987, 'height': 0.28167659027887154} 
        length_till_shift, shift_length = combining_routes.find_shift_length(
            tracks, path
        )
        self.assertTrue(length_till_shift>=0)
        #print(length_till_shift)
        #print(shift_length)

    def test10_find_shift_length(self):
        tracks = [
            {'length': 0.0019061394364962218, 'height': -0.2230461360862837},
            {'length': 0.001280597461137607, 'height': -0.17942790734837177},
            {'length': 0.003547419439255187, 'height': -0.16684866975505908},
            {'length': 0.0026727699269563964, 'height': -0.1514126857047556},
            {'length': 0.0005930737361545882, 'height': -0.16761378492166606}
        ]
        path = {'length': 0.004501471916353294, 'height': -0.19081592394007982}
        length_before_shift, shift_length = combining_routes.find_shift_length(
            tracks, path
        )
        #print(length_before_shift)
        self.assertEqual(shift_length, path["length"])

    def test1_find_optimum_list(self):
        tracks = [
            {"length":1.0, "height": 2},
            {"length":0.9, "height": 1.6},
            {"length":0.1, "height": 3.0},
            {"length":0.5, "height": 2.2},
            {"length":2.0, "height": 3.2}
        ]
        path = {"length":1.5, "height":4}
        opt_list = combining_routes.find_optimum_list([], 4, tracks, path)
        print(opt_list)
        expected_opt_list = [
            {"length":1.5, "height":4},
            {"length":1.0, "height":2},
            {"length":0.9, "height":1.6},
            {"length":0.1, "height":3.0},
            {"length":0.5, "height":2.2},
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

    def test2_find_optimum_list(self):
        tracks = [
            {"length":1.0, "height": 2},
            {"length":0.9, "height": 1.6},
            {"length":0.1, "height": 2.9},
            {"length":1.0, "height": 2.2},
            {"length":2.0, "height": 3.2}
        ]
        path = {"length":1.5, "height":3}
        threshold = 3
        opt_list = combining_routes.find_optimum_list([], threshold, tracks, path)
        print(opt_list)
        expected_opt_list = [
            {"length":1.5, "height":3},
            {"length":1.0, "height":2},
            {"length":0.5, "height":1.6}
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])
       
    def test3_find_optimum_list(self):
        tracks = [
            {"length":1.0, "height": 2},
            {"length":0.9, "height": 1.6},
            {"length":0.1, "height": 3.0},
            {"length":0.5, "height": 2.2},
            {"length":2.0, "height": 3.2}
        ]
        path = {"length":1.5, "height":1}
        threshold = 3.5
        old_opt_list = [{"length":2 , "height":0.5}]
        opt_list = combining_routes.find_optimum_list(
            old_opt_list, threshold, tracks, path
        )
        print(opt_list)
        expected_opt_list = [
            {"length":2, "height":0.5},
            {"length":1, "height":2},
            {"length":0.9, "height":1.6},
            {"length":0.1, "height":3.0},
            {"length":0.5, "height":2.2},
            {"length":1, "height":3.2}
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

    def test4_find_optimum_list(self):
        tracks = []
        path = {"length":1.5, "height":1}
        threshold = 3.5
        old_opt_list = [{"length":2 , "height":0.5}]
        opt_list = combining_routes.find_optimum_list(
            old_opt_list, threshold, tracks, path
        )
        print(opt_list)
        expected_opt_list = [
            {"length":2, "height":0.5},
            {"length":1.5, "height":1}
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

    def test5_find_optimum_list(self):
        tracks = []
        path = {"length":1.5, "height":1}
        threshold = 1 
        old_opt_list = [{"length":2 , "height":0.5}]
        opt_list = combining_routes.find_optimum_list(
            old_opt_list, threshold, tracks, path
        )
        print(opt_list)
        expected_opt_list = [
            {"length":2, "height":0.5},
            {"length":1, "height":1}
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

    def test6_find_optimum_list(self):
        tracks = [
            {"length":1.0, "height": 3},
            {"length":0.9, "height": 2.6},
            {"length":0.1, "height": 1.0},
            {"length":0.5, "height": 2.2},
            {"length":2.0, "height": 3.2}
        ]
        path = {"length":1.5, "height":2.5}
        threshold = 4.0
        old_opt_list = []
        opt_list = combining_routes.find_optimum_list(
            old_opt_list, threshold, tracks, path
        )
        print(opt_list)
        expected_opt_list = [
            {"length":1, "height":3},
            {"length":0.9, "height":2.6},
            {"length":0.6 + 0.3/0.7, "height":2.5},
            {"length":1.07142857, "height":3.2}
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

    def test7_find_optimum_list(self):
        tracks = [
            {"length":0.5, "height": 1.8},
            {"length":1, "height": 1},
            {"length":1, "height": 2.5},
            {"length":0.5, "height": 1.8},
            {"length":2.0, "height": 3}
        ]
        path = {"length":2, "height":2}
        threshold = 4.6
        old_opt_list = []
        opt_list = combining_routes.find_optimum_list(
            old_opt_list, threshold, tracks, path
        )
        print(opt_list)
        expected_opt_list = [
            {"length":2, "height":2},
            {"length":0.5, "height":1.8},
            {"length":0.5, "height":1},
            {"length":0.1, "height":2},
            {"length":1.5, "height":3}
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

    def test8_find_optimum_list(self):
        tracks = [
            {"length":0.5, "height": 1.8},
            {"length":1, "height": 1},
            {"length":1, "height": 2.5},
            {"length":2, "height": 1.8},
            {"length":1.0, "height": 1.9}
        ]
        path = {"length":2, "height":2}
        threshold = 6
        old_opt_list = []
        opt_list = combining_routes.find_optimum_list(
            old_opt_list, threshold, tracks, path
        )
        print(opt_list)
        expected_opt_list = [
            {"length":2, "height":2},
            {"length":0.5, "height":1.8},
            {"length":0.5, "height":1},
            {"length":1.5, "height":2},
            {"length":1.5, "height":1.8}
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

    def test9_find_optimum_list(self):
        tracks = [
            {'length': 0.0019061394364962218, 'height': -0.22304613608628371}, 
            {'length': 0.0012805974611376069, 'height': -0.17942790734837177},
            {'length': 0.0035474194392551869, 'height': -0.16684866975505908},
            {'length': 0.0026727699269563964, 'height': -0.15141268570475561},
            {'length': 0.00059307373615458819, 'height': -0.16761378492166606}
        ]
        path = {'length': 0.0045014719163532944, 'height': -0.19081592394007982}
        threshold = 0.01
        opt_route = combining_routes.find_optimum_list(
            [], threshold, tracks, path
        )

    def test10_find_optimum_list(self):
        threshold = 0.01
        tracks = [
            {'length': 0.00084789403180791879, 'height': 0.23448309516230831},
            {'length': 0.0012285759186611484, 'height': 0.23324036207975657},
            {'length': 0.0019266030665390258, 'height': 0.2330327242277965},
            {'length': 9.4296078588858301e-05, 'height': 0.23492283715174436},
            {'length': 0.00031277006708566989, 'height': 0.22411516857094832},
            {'length': 4.1058012621537562e-06, 'height': 0.22339328159307462},
            {'length': 0.0013797741130892873, 'height': 0.22788296716113143},
            {'length': 0.0028259705331712612, 'height': 0.22378119371125113},
            {'length': 0.00052765220697655046, 'height': 0.22003631481562122},
            {'length': 0.00077062538463392935, 'height': 0.21754966007241372},
            {'length': 8.1732798184196848e-05, 'height': 0.21573718945302311}
        ]
        path = {'length': 0.00084789403180791879, 'height': 0.23448309516230831}
        opt_route = combining_routes.find_optimum_list(
            [], threshold, tracks, path
        )

    def test11_find_optimum_list(self):
        tracks = [
            {'length': 0.00025062494574424579, 'height': 0.5298658136278781},
            {'length': 0.0014941893570154375, 'height': 0.53781981860961281},
            {'length': 0.00061166176929072107, 'height': 0.50414035305944505},
            {'length': 0.002477240484939346, 'height': 0.51011589430167237},
            {'length': 0.0028098073709598445, 'height': 0.5250412056705448}
        ]
        path = {'length': 0.002477240484939346, 'height': 0.51011589430167237}
        threshold = 0.00764352392795
        opt_route = combining_routes.find_optimum_list(
            [], threshold, tracks, path
        )

    def test12_find_optimum_list(self):
        tracks = [
            {'length': 0.0015818624519766198, 'height': 0.26080389330262954},
            {'length': 0.0023895863196095133, 'height': 0.26350772901313507},
            {'length': 0.001919040275049572, 'height': 0.26721284616747887},
            {'length': 0.00063398722027372106, 'height': 0.27249704751316284},
            {'length': 0.00090645013618520249, 'height': 0.26845843688159043},
            {'length': 0.001713917479763888, 'height': 0.26945861543539862},
            {'length': 0.00034247804054427997, 'height': 0.27422459923821024}
        ]
        path = {'length': 0.0029284872016562035, 'height': 0.26906053606224278}
        threshold = 0.0094873219234
        opt_route = combining_routes.find_optimum_list(
            [], threshold, tracks, path
        )

    def test1_combine_routes(self):
        opt_route = [
            {"length":1.5, "height": 2.8},
            {"length":1, "height": 3},
            {"length":1, "height": 4.5},
        ]
        new_route = [
            {"length":0.5, "height": 1.8},
            {"length":1, "height": 1},
            {"length":2.0, "height": 1.9}
        ]
        threshold = 3.5 
        opt_list = combining_routes.combine_routes(opt_route, new_route, threshold)
        expected_opt_list = [
            {"length":1.5, "height": 2.8},
            {"length":1, "height": 3},
            {"length":1, "height": 4.5}
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

    def test2_combine_routes(self):
        opt_route = [
            {"length":1.5, "height": 3.8},
            {"length":1, "height": 3},
            {"length":1, "height": 2.5},
        ]
        new_route = [
            {"length":0.5, "height": 2.9},
            {"length":1, "height": 1},
            {"length":2.0, "height": 1.9}
        ]
        threshold = 3.5 
        opt_list = combining_routes.combine_routes(opt_route, new_route, threshold)
        print(opt_list)
        expected_opt_list = [
            {"length":1.5, "height": 3.8},
            {"length":1, "height": 3},
            {"length":0.5, "height": 2.9},
            {"length":0.5, "height": 2.5}
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

    def test3_combine_routes(self):
        opt_route = [
            {"length":1.5, "height": 3.8},
            {"length":1, "height": 3},
            {"length":1, "height": 2.5},
        ]
        new_route = [
            {"length":0.5, "height": 2.9},
            {"length":0.05, "height": 2},
            {"length":2.95, "height": 2.7}
        ]
        threshold = 3.5 
        opt_list = combining_routes.combine_routes(opt_route, new_route, threshold)
        print(opt_list)
        expected_opt_list = [
            {"length":1.5, "height": 3.8},
            {"length":1, "height": 3},
            {"length":0.5, "height": 2.9},
            {"length":0.175, "height": 2.5},
            {"length":0.325, "height": 2.7}
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

    def test4_combine_routes(self):
        opt_route = [
            {"length":1.5, "height": 3.8},
            {"length":1, "height": 3},
            {"length":1, "height": 2.5},
        ]
        new_route = [
            {"length":0.5, "height": 3.9},
            {"length":0.05, "height": 2},
            {"length":2.95, "height": 2.7}
        ]
        threshold = 3.5 
        opt_list = combining_routes.combine_routes(opt_route, new_route, threshold)
        print("opt list {}".format(opt_list))
        expected_opt_list = [
            {"length":0.5, "height": 3.9},
            {"length":1.5, "height": 3.8},
            {"length":1, "height": 3},
            {"length":0.175, "height": 2.5},
            {"length":0.325, "height": 2.7}
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

    def test5_combine_routes(self):
        opt_route = [
            {"length":2, "height": 3},
            {"length":1, "height": 4},
            {"length":1.5, "height": 2.5},
            {"length":1, "height": 1.5},
            {"length":3, "height": 5},
            {"length":4, "height": 1},
        ]
        new_route = [
            {"length":1, "height": 2},
            {"length":2, "height": 2.5},
            {"length":3, "height": 3},
            {"length":2, "height": 3.5},
            {"length":0.5, "height": 3.3},
            {"length":4, "height": 3},
        ]
        threshold = 12.5 
        opt_list = combining_routes.combine_routes(opt_route, new_route, threshold)
        print("opt list {}".format(opt_list))
        expected_opt_list = [
            {"length":2, "height": 3},
            {"length":1, "height": 4},
            {"length":1.5, "height": 2.5},
            {"length":1, "height": 2},
            {"length":0.2, "height": 2.5},
            {"length":2.8, "height": 5},
            {"length":1, "height": 2},
            {"length":2, "height": 2.5},
            {"length":1, "height": 3},
        ]
        for count, track in enumerate(expected_opt_list):
            self.assertAlmostEqual(track["length"], opt_list[count]["length"])
            self.assertAlmostEqual(track["height"], opt_list[count]["height"])

