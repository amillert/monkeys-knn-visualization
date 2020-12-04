from utils import check_hexacolor


class Monkey:
    def __init__(self, fur_color: str, size: float, weight: float, species: str = ""):
        if not check_hexacolor(fur_color): raise ValueError
        else: self.fur_color = fur_color
        self.size = size
        self.weight = weight
        self.species = species

    def __str__(self):
        return f"Monkey(fur: {self.fur_color}; size: {self.size}; weight: {self.weight}; species: {self.species})"

    def __repr__(self):
        return self.__str__()

    def compute_bmi(self):
        return self.weight / self.size / self.size

    @staticmethod
    def monkify(row):
        # shouldn't Monkey get int color and bmi?
        return Monkey(row["color"], row["size"], row["weight"], row["species"])

