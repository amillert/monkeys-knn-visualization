from monkey_model import Monkey
import utils

from collections import Counter
import pandas as pd
pd.options.display.width = 0


def preprocess(df, cols):
    df.dropna(subset=[col for col in cols if col != "species"], inplace=True)
    df = df[df["color"].map(utils.check_hexacolor)]
    df = df[df["size"].map(lambda l: l > 0.0)]
    df = df[df["weight"].map(lambda l: l > 0.0)]
    df["monkey"] = df[cols].apply(Monkey.monkify, axis=1)
    df["fur_color_int"] = df.color.map(lambda l: int(l[1:], 16))
    df["bmi"] = df.monkey.map(lambda l: l.compute_bmi())
    return df


def read_monkeys_from_csv(csv_path):
    df = pd.read_csv(csv_path, header=0)
    cols = ["color", "size", "weight", "species"]
    expected_cols = set(cols)
    if set(df.columns) != expected_cols:
        raise ValueError
    df = df.astype({"color": str, "size": float, "weight": float, "species": str})
    return preprocess(df, cols)


def species_split(df):
    return df[df.species == "nan"].copy(), df[df.species != "nan"].copy()


def update_species(m, s):
    m.species = s
    return m


def choose_species(single_dimensions_row, all_dimensions_monkeys, k):
    lol = map(lambda l: (utils.euclidean_distance(single_dimensions_row, l[0]), l[0], l[1]), all_dimensions_monkeys)
    # lol = [(euclidean_distance(point, p), p, m) for p, m in all_points_monkeys]

    # returns closest monkeys
    species = map(lambda l: l[2].species, sorted(lol, key=lambda x: x[0])[:k])
    return sorted(Counter(species).items(), key=lambda l: -l[1])[0][0]


def compute_knn(df, k=10, dimensions=None):
    """
        size, weight, BMI, fur color, fur color red value,
        fur color blue value, fur color green value and fur color intensity
    """
    # only for 2D...
    # df["point"] = df[["bmi", "fur_color_int"]].apply(Point.pointify, axis=1)
    dimensions = ["bmi", "fur_color_int"] if not dimensions else dimensions
    if "R" in dimensions or "G" in dimensions or "B" in dimensions:
        df["R"], df["G"], df["B"] = zip(*df["color"].apply(utils.getRGBChannels2int))

    X_empty, X = species_split(df)

    X_empty["dimensions"] = X_empty[dimensions].apply(lambda l: tuple(l), axis=1)

    # points_non_empty = list(X.point.values.reshape(1, -1)[0])
    points_non_empty = [tuple(x) for x in X[dimensions].values]
    # print(points_non_empty)

    # print(X_empty[dimensions].values.reshape(len(dimensions), -1)[:3])
    # print(X_empty[dimensions].values.reshape(1, -1)[:3])
    # print(len(X_empty[dimensions].values.reshape(1, -1)[:3]))
    # print(len(X_empty[dimensions].values.reshape(len(, -1)))
    # exit(11)

    # monkeys_non_empty = list(X.monkey.values.reshape(1, -1)[0])
    monkeys_non_empty = list(X.monkey.values)

    non_empty = [*zip(points_non_empty, monkeys_non_empty)]

    X_empty["species"] = X_empty.dimensions.apply(lambda l: choose_species(l, non_empty, k))

    # not asked for it, so make sure it's desired
    # X_empty["monkey"] = X_empty[["monkey", "species"]].apply(lambda l: update_species(l[0], l[1]), axis=1)

    # cols = [c for c in X.columns if c != "point"]
    cols = [*filter(lambda l: l != "point", X.columns)]
    return pd.concat([X[cols], X_empty[cols]]).reset_index(drop=True, inplace=False)


def save_to_csv(df, csv_filename):
    df[["species", "color", "size", "weight"]].to_csv(csv_filename, ",")


def main():
    args = utils.get_cli_args()

    try:
        if args.dims and len(args.dims) < 2:
            raise ValueError
    except ValueError:
        print(f"Too few dimensions provided; at least 2 required!")
    else:
        df = read_monkeys_from_csv(args.in_path)
        print(f"Shape before joining: {df.shape}")

        joined_df = compute_knn(df)
        print(f"Shape after joining: {joined_df.shape}")

        save_to_csv(joined_df, args.out_path)
        # xd = pd.read_csv("out.csv", index_col=0); print(xd)


if __name__ == "__main__":
    main()
