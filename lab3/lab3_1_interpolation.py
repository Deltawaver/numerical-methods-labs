import math
import sys
from pathlib import Path

import numpy as np


def lagrange(x, y, point):
    return sum(
        yi * math.prod((point - xj) / (xi - xj)
                       for j, xj in enumerate(x) if j != i)
        for i, (xi, yi) in enumerate(zip(x, y))
    )


def newton_coefficients(x, y):
    coefficients = np.array(y, dtype=float)
    for order in range(1, len(x)):
        coefficients[order:] = (
            (coefficients[order:] - coefficients[order - 1:-1])
            / (x[order:] - x[:-order])
        )
    return coefficients


def newton_value(x, coefficients, point):
    value = coefficients[-1]
    for i in range(len(coefficients) - 2, -1, -1):
        value = coefficients[i] + (point - x[i]) * value
    return value


def main():
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("data3.1.txt")
    rows = filename.read_text(encoding="utf-8-sig").splitlines()
    point = float(rows[0])
    exact = math.asin(point)
    print(f"\n3.1. Интерполяция, arcsin({point}) = {exact:.10f}")
    for name, nodes in zip(("а", "б"), rows[1:]):
        x = np.fromstring(nodes, sep=" ")
        y = np.arcsin(x)
        weights = [yi / math.prod(xi - xj for j, xj in enumerate(x) if j != i)
                   for i, (xi, yi) in enumerate(zip(x, y))]
        coefficients = newton_coefficients(x, y)
        print(f"{name}) узлы {x}")
        print("Коэффициенты при произведениях Лагранжа:", np.array(weights))
        print("Разделённые разности Ньютона:", coefficients)
        for method, value in [("Лагранж", lagrange(x, y, point)),
                              ("Ньютон", newton_value(x, coefficients, point))]:
            print(f"{method}: {value:.10f}, абсолютная ошибка {abs(value - exact):.10g}")


if __name__ == "__main__":
    main()
