from utils import check_hexacolor

import pandas as pd


class Monkey:
    def __init__(self, fur_color: str, size: float, weight: float, species: str = "") -> None:
        if not check_hexacolor(fur_color):
            raise ValueError
        else:
            self.fur_color = fur_color
        self.size = size
        self.weight = weight
        self.species = species

    def __str__(self) -> str:
        return f"Monkey(fur: {self.fur_color}; size: {self.size}; weight: {self.weight}; species: {self.species})"

    def __repr__(self) -> str:
        return self.__str__()

    def compute_bmi(self) -> float:
        return self.weight / self.size / self.size

    @classmethod
    def monkify(cls, row: pd.Series) -> object:
        try:
            return cls(row["color"], row["size"], row["weight"], row["species"])
        except ValueError:
            print("Hex color is not of the correct format!")

