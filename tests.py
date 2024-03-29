from monkey_model import Monkey
import monkey_classify as mc
import utils as utl

import pandas as pd
import unittest

pd.set_option('display.max_colwidth', None)


class TestMonkey(unittest.TestCase):
    m1 = Monkey("#AB1234", 12, 40.5)
    m2 = Monkey("#AB1234", 12, 40.5, "gorilla")
    base_cols = ["color", "size", "weight", "species"]
    save_cols = ["fur_color_int", "size", "weight", "species"]

    def test_monkey_str(self):
        expected_m1 = f"Monkey(fur: #AB1234; size: 12; weight: 40.5; species: )"
        expected_m2 = f"Monkey(fur: #AB1234; size: 12; weight: 40.5; species: gorilla)"
        self.assertEqual(str(self.m1), expected_m1)
        self.assertEqual(str(self.m2), expected_m2)
        self.assertNotEqual(str(self.m1), expected_m2)

    def test_monkey_color_exception(self):
        # 'G' not hex
        with self.assertRaises(ValueError):
            Monkey("#AG1234", 12, 40.5)

        # to many values per color
        with self.assertRaises(ValueError):
            Monkey("#AF12345", 12, 40.5)

        # mixed hex format
        with self.assertRaises(ValueError):
            Monkey("#Af1234", 12, 40.5)

    def test_colors(self):
        color = "#AF1234"
        self.assertEqual(
            (utl.getRfromHex2int(color),
             utl.getGfromHex2int(color),
             utl.getBfromHex2int(color)),
            utl.getRGBChannels2int(color)
        )

        color = "#FFFFFF"
        self.assertEqual((255, 255, 255), utl.getRGBChannels2int(color))  # actually helped me detect a bug ;)

    def test_monkey_bmi(self):
        self.assertEqual(self.m1.compute_bmi(), self.m1.weight/self.m1.size/self.m1.size)

    def test_read_monkeys_csv(self):
        expected_cols = {"color", "size", "weight", "species"}

        with self.assertRaises(ValueError):
            mc.read_monkeys_from_csv("./monkeys_wrong_cols_empty.csv")

        read_cols = set(mc.read_monkeys_from_csv("./monkeys.csv").columns)
        self.assertEqual(expected_cols, read_cols)

    def test_read_monkeys_csv_strict(self):
        # why do tests in read_monkeys for errors require with, while test_argparser doesn't?
        with self.assertRaises(ValueError):
            mc.read_monkeys_from_csv("./monkeys.csv", strict=True)

    def test_df_preprocessing(self):
        data_pre = [
            ["#aaaaaa", 0.14, 14.5, "gorilla"],
            ["#aaaaaa", None, 14.5, "gorilla"],
            ["#aaaaaa", 0.14, None, "gorilla"],
            ["#aaaa", 0.14, 14.5, "gorilla"],
            [None, 0.14, 14.5, "gorilla"],  # so far only first will remain
            ["#aaaaaa", 0.27, 16.3, None],  # this one remains too
            ["#123456", 1.4, 12.0, "patafian"],
            ["#123456", 1.8, 10.0, "patafian"]
        ]
        df_pre = pd.DataFrame(columns=self.base_cols, data=data_pre)

        data_post = [
            ["#aaaaaa", 0.14, 14.5, "gorilla", Monkey("#aaaaaa", 0.14, 14.5, "gorilla"), 11184810, 14.5/0.14/0.14],
            ["#aaaaaa", 0.27, 16.3, None, Monkey("#aaaaaa", 0.27, 16.3, None), 11184810, 16.3/0.27/0.27],
            ["#123456", 1.4, 12.0, "patafian", Monkey("#123456", 1.4, 12.0, "patafian"), 1193046, 12.0/1.4/1.4],
            ["#123456", 1.8, 10.0, "patafian", Monkey("#123456", 1.8, 10.0, "patafian"), 1193046, 10.0/1.8/1.8]
        ]
        df_post_ref = pd.DataFrame(columns=self.base_cols + ["monkey", "fur_color_int", "bmi"], data=data_post)

        df_post = mc.preprocess(df_pre)

        self.assertEqual(df_post.shape, df_post_ref.shape)
        self.assertEqual({str(x) for x in df_post.monkey} & {str(x) for x in df_post_ref.monkey},
                         {str(x) for x in df_post.monkey})
        self.assertEqual([str(x) for x in df_post.bmi], [str(x) for x in df_post_ref.bmi])
        self.assertEqual([str(x) for x in df_post.fur_color_int], [str(x) for x in df_post_ref.fur_color_int])

    def test_save_with_read_csv(self):
        tmp_path = ".tmp.csv"

        df = pd.DataFrame([[0, 1.0, 2.0, "3"], [4, 5.0, 6.0, "7"]], columns=self.save_cols)
        mc.save_to_csv(df, tmp_path)
        dff = mc.read_monkeys_from_csv(tmp_path)
        self.assertEqual(df.shape, dff.shape)
        for col in self.save_cols:
            self.assertEqual(list(df[col].values), list(dff[col].values))
        import os
        os.remove(tmp_path)

    def test_euclidean_distance(self):
        p2D = [0.3, 0.5]
        q2D = [1.3, 1.2]
        self.assertAlmostEqual(utl.euclidean_distance(p2D, q2D), 1.220656, 6)

        p3D = [0.3, 0.5, 0.8]
        q3D = [1.3, 0.4, 1.2]
        self.assertAlmostEqual(utl.euclidean_distance(p3D, q3D), 1.081665, 6)

        p0 = [0.0] * 40
        q0 = [0.0] * 40
        self.assertEqual(utl.euclidean_distance(p0, q0), 0)

    def test_knn(self):
        df_empty_init = [
            ["#d75820", 1.342, 84.632, "nan"],
            ["#c65622", 0.759, 47.952, "nan"],
            ["#c65622", 1.474, 51.925, "nan"],
            ["#111911", 0.710, 46.961, "nan"]
        ]

        df_nonempty_init = [
            ["#2e1628", 1.012, 46.063, "orangutan"],
            ["#210180", 1.657, 212.974, "gorilla"],
            ["#ce69ff", 1.008, 47.306, "orangutan"],
            ["#12211a", 0.770, 45.306, "bonobo"]
        ]

        df = pd.DataFrame(df_nonempty_init + df_empty_init, columns=self.base_cols)
        preprocessed = mc.preprocess(df)
        knned = mc.compute_knn(preprocessed, 3)

        self.assertEqual(preprocessed.shape, knned.shape)
        self.assertEqual(
            list(knned.species),
            ["orangutan", "gorilla", "orangutan", "bonobo", "orangutan", "orangutan", "orangutan", "bonobo"]
        )

        knned2 = mc.compute_knn(preprocessed, 3, ["R", "B"])
        self.assertEqual(
            list(knned2.species),
            ["orangutan", "gorilla", "orangutan"] + ["bonobo"] * 5
        )

    def test_argparser(self):
        # from argparse import ArgumentError

        args = ["knn", '-i', 'monkeys.csv', '-o', 'out.csv', '-d', 'bmi', 'fur_color_int', 'weight']
        parsed, sub = utl.get_cli_args(args)
        self.assertEqual(sub, "knn")
        self.assertTrue(parsed.in_path)
        self.assertTrue(parsed.out_path)
        self.assertTrue(parsed.dims)

        args = ["knn", "-i", "monkeys.csv", "-o", "out.csv", "-d", "bmi"]
        parsed, sub = utl.get_cli_args(args)
        self.assertRaises(Exception, parsed.dims)

        args = ["visual", '-i', 'out.csv', '-f', 'size', 'fur_color_int']
        parsed, sub = utl.get_cli_args(args)
        self.assertEqual(sub, "visual")
        self.assertTrue(parsed.in_path)
        self.assertTrue(parsed.features)
        self.assertTrue(len(parsed.features))

        # TODO: why does it not work? Because it's already handled?
        # args = ["visual", '-i', 'out.csv', '-f', 'size']  # valid, but too few
        # self.assertRaises(ArgumentError, utl.get_cli_args(args))

        # args = ["visual", '-i', 'out.csv', '-f', 'weight', 'fur_color_int', 'size']  # all valid but too many
        # self.assertRaises(ArgumentError, utl.get_cli_args(args))

        # args = ["visual", '-i', 'out.csv', '-f', 'weight', 'size', 'bmi']  # enough, but not from list
        # self.assertRaises(ArgumentError, utl.get_cli_args(args))

        # args = ["knn", "-i", "monkeys.csv", "-o", "out.csv", "-d", "bmi", "coś_tam"]
        # import argparse
        # self.assertRaises(ArgumentError, utl.get_cli_args(args).dims)


if __name__ == "__main__":
    args = ["knn", "-i", "monkeys.csv", "-o", "out.csv", "-d", "bmi", "fur_color_int", "weight"]
    unittest.main(argv=args)
