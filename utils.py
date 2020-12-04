import argparse
import math
import re


def check_hexacolor(color_str: str) -> bool:
    reg = r"#([0-9a-f]{6}|[0-9A-F]{6})\b"  # let's disallow mixed format
    # of course it's no biggie to allow it; but I just don't want ot allow it -> r"[0-9a-fA-F]"
    return bool(re.match(reg, color_str))


def hex2int(s: str) -> int:
    return int(s, 16)


def getRfromHex2int(s: str) -> int:
    return hex2int(s[2:4])


def getGfromHex2int(s: str) -> int:
    return hex2int(s[4:6])


def getBfromHex2int(s: str) -> int:
    return hex2int(s[6:])


def getRGBChannels2int(s: str) -> (int, int, int):
    return getRfromHex2int(s), getGfromHex2int(s), getBfromHex2int(s)


def euclidean_distance(p, q):
    return math.sqrt(sum([(p_i - q_i) ** 2 for p_i, q_i in zip(p, q)]))


def get_cli_args():
    argument_parser = argparse.ArgumentParser(
        prog="monkeys",
        description="Argument parser of the Monkeys' project",
        epilog="Compute KNNs for Monkeys to predict unknown species",
        allow_abbrev=True
    )
    argument_parser.version = "0.1"
    argument_parser.add_argument(
        "-i",
        "--in_path",
        action="store",
        type=str,
        help="Provide an input csv file to read monkeys",
        required=True,
    )
    argument_parser.add_argument(
        "-o",
        "--out_path",
        action="store",
        type=str,
        help="Provide an output csv file to save updated monkeys",
        required=True,
    )
    argument_parser.add_argument(
        "-d",
        "--dims",
        action="store",
        type=str,
        choices=["bmi", "fur_color_int", "R", "G", "B", "weight"],
        help="Provide a list of attributes to utilize in the KNN",
        nargs="+"
    )

    return argument_parser.parse_args()
