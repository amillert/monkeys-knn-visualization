#!/usr/bin/env python

from monkey_model import Monkey
from monkey_visualize import scatter_plot
import utils

from collections import Counter
import pandas as pd
pd.options.display.width = 0

COLS = ["color", "size", "weight", "species"]
OUT_COLS = ["fur_color_int", "size", "weight", "species"]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=[col for col in COLS if col != "species"], inplace=True)
    df = df[df["color"].map(utils.check_hexacolor)]
    df = df[df["size"].map(lambda l: l > 0.0)]
    df = df[df["weight"].map(lambda l: l > 0.0)]
    df["monkey"] = df[COLS].apply(Monkey.monkify, axis=1)
    df["fur_color_int"] = df.color.map(lambda l: utils.hex2int(l[1:]))
    df["bmi"] = df.monkey.map(lambda l: l.compute_bmi())
    return df


def read_monkeys_from_csv(csv_path: str, strict: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=0)
    if strict:
        if df.shape != df.dropna(inplace=False).shape:
            raise ValueError
    elif not set(df.columns.to_list()) <= set(COLS) | set(OUT_COLS):
        raise ValueError

    if "color" in df.columns:
        df = df.astype({"species": str, "size": float, "weight": float, "color": str})
    elif "fur_color_int" in df.columns:
        df = df.astype({"species": str, "size": float, "weight": float, "fur_color_int": int})
    return df


def species_split(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    return df[df.species == "nan"].copy(), df[df.species != "nan"].copy()


def update_species(m: Monkey, s: str) -> Monkey:
    m.species = s
    return m


def choose_species(single_dimensions_row: pd.Series, all_dimensions_monkeys: list, k: int) -> str:
    lol = map(lambda l: (utils.euclidean_distance(single_dimensions_row, l[0]), l[0], l[1]), all_dimensions_monkeys)

    # returns closest monkeys
    species = map(lambda l: l[2].species, sorted(lol, key=lambda x: x[0])[:k])
    return sorted(Counter(species).items(), key=lambda l: -l[1])[0][0]


def compute_knn(df: pd.DataFrame, k: int = 10, dimensions: list = None) -> pd.DataFrame:
    dimensions = ["bmi", "fur_color_int"] if not dimensions else dimensions
    if "R" in dimensions or "G" in dimensions or "B" in dimensions:
        df["R"], df["G"], df["B"] = zip(*df["color"].apply(utils.getRGBChannels2int))

    X_empty, X = species_split(df)

    X_empty["dimensions"] = X_empty[dimensions].apply(lambda l: tuple(l), axis=1)

    points_non_empty = [tuple(x) for x in X[dimensions].values]
    monkeys_non_empty = list(X.monkey.values)

    non_empty = [*zip(points_non_empty, monkeys_non_empty)]

    X_empty["species"] = X_empty.dimensions.apply(lambda l: choose_species(l, non_empty, k))

    cols = [*filter(lambda l: l != "point", X.columns)]
    return pd.concat([X[cols], X_empty[cols]]).reset_index(drop=True, inplace=False)


def save_to_csv(df: pd.DataFrame, csv_filename: str) -> None:
    df[OUT_COLS].to_csv(csv_filename, sep=",", header=True, index=False)


def main() -> None:
    args, subparser = utils.get_cli_args()

    try:
        if subparser == "knn" and args.dims and len(args.dims) < 2:
            raise ValueError
    except ValueError:
        print(f"Too few dimensions provided; at least 2 required!")
    else:
        if subparser == "knn":
            df = preprocess(read_monkeys_from_csv(args.in_path))
            print(f"Shape before joining: {df.shape}")

            joined_df = compute_knn(df)
            print(f"Shape after joining: {joined_df.shape}")

            save_to_csv(joined_df, args.out_path)
        elif subparser == "visual":
            try:
                df = read_monkeys_from_csv(args.in_path, strict=True)
            except ValueError:
                print("Found nan values")
            else:
                X, Y = list(df[args.features[0]].values), list(df[args.features[1]].values)
                labels = list(df["species"].values)
                scatter_plot(X, Y, labels)


if __name__ == "__main__":
    main()
