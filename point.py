class Point:
    def __init__(self, bmi, color):
        self.x = bmi
        self.y = color

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    @classmethod
    def pointify(cls, row):
        return cls(row["bmi"], row["fur_color_int"])




