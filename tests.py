from monkey_model import Monkey
from monkey_classify import read_monkeys_from_csv as read
from monkey_classify import preprocess

import pandas as pd
import unittest

pd.set_option('display.max_colwidth', None)


class TestMonkey(unittest.TestCase):
    m1 = Monkey("#AB1234", 12, 40.5)
    m2 = Monkey("#AB1234", 12, 40.5, "gorilla")

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

    def test_monkey_bmi(self):
        self.assertEqual(self.m1.compute_bmi(), self.m1.weight/self.m1.size/self.m1.size)

    def test_read_monkeys_csv(self):
        expected_cols = {"color", "size", "weight", "species", "monkey", "fur_color_int", "bmi"}

        with self.assertRaises(ValueError):
            read("./monkeys_wrong_cols_empty.csv")

        read_cols = set(read("./monkeys.csv").columns)
        self.assertEqual(expected_cols, read_cols)

    def test_df_preprocessing(self):
        cols = ["color", "size", "weight", "species"]
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
        df_pre = pd.DataFrame(columns=cols, data=data_pre)

        data_post = [
            ["#aaaaaa", 0.14, 14.5, "gorilla", Monkey("#aaaaaa", 0.14, 14.5, "gorilla"), 11184810, 14.5/0.14/0.14],
            ["#aaaaaa", 0.27, 16.3, None, Monkey("#aaaaaa", 0.27, 16.3, None), 11184810, 16.3/0.27/0.27],
            ["#123456", 1.4, 12.0, "patafian", Monkey("#123456", 1.4, 12.0, "patafian"), 1193046, 12.0/1.4/1.4],
            ["#123456", 1.8, 10.0, "patafian", Monkey("#123456", 1.8, 10.0, "patafian"), 1193046, 10.0/1.8/1.8]
        ]
        df_post_ref = pd.DataFrame(columns=cols + ["monkey", "fur_color_int", "bmi"], data=data_post)

        df_post = preprocess(df_pre, cols)

        self.assertEqual(df_post.shape, df_post_ref.shape)
        self.assertEqual({str(x) for x in df_post.monkey} & {str(x) for x in df_post_ref.monkey}, {str(x) for x in df_post.monkey})
        self.assertEqual([str(x) for x in df_post.bmi], [str(x) for x in df_post_ref.bmi])
        self.assertEqual([str(x) for x in df_post.fur_color_int], [str(x) for x in df_post_ref.fur_color_int])

    # TODO 1: tests for euclidean
    # TODO 2: tests for knn

    def test_argparser(self):
        from utils import args
        self.assertIsInstance(args.in_path[0], str)


if __name__ == "__main__":
    unittest.main()
