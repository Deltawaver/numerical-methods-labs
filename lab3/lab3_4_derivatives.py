import sys
from pathlib import Path

import numpy as np


def newton_coefficients(x, y):
    coefficients = np.array(y, dtype=float)
    for order in range(1, len(x)):
        coefficients[order:] = (
            (coefficients[order:] - coefficients[order - 1:-1])
            / (x[order:] - x[:-order])
        )
    return coefficients


def derivatives(x, y, point):
    # Use three adjacent nodes, centered at an interior knot when possible.
    start = int(np.clip(np.searchsorted(x, point) - 1, 0, len(x) - 3))
    nodes = x[start:start + 3]
    _, b, c = newton_coefficients(nodes, y[start:start + 3])
    return b + c * (2 * point - nodes[0] - nodes[1]), 2 * c


def main():
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("data3.4.txt")
    rows = filename.read_text(encoding="utf-8-sig").splitlines()
    print("\n3.4. Численное дифференцирование")
    point = float(rows[0])
    x, y = (np.fromstring(row, sep=" ") for row in rows[1:])
    first, second = derivatives(x, y, point)
    print(f"y'({point:g}) = {first:.10f}, y''({point:g}) = {second:.10f}")


if __name__ == "__main__":
    main()
