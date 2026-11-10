import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab1.progonka import progonka


def natural_spline(x, y):
    """Coefficients of each cubic in ascending powers of x - x[i]."""
    h = np.diff(x)
    slopes = np.diff(y) / h
    diagonal = 2 * (h[:-1] + h[1:])
    c = np.zeros(len(x))
    if len(diagonal):
        c[1:-1], _, _ = progonka(
            np.r_[0, h[1:-1]], diagonal, np.r_[h[1:-1], 0],
            3 * np.diff(slopes), len(diagonal),
        )
    b = slopes - h * (2 * c[:-1] + c[1:]) / 3
    d = np.diff(c) / (3 * h)
    return np.column_stack((y[:-1], b, c[:-1], d))


def main():
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("data3.2.txt")
    rows = filename.read_text(encoding="utf-8-sig").splitlines()
    point = float(rows[0])
    x, y = (np.fromstring(row, sep=" ") for row in rows[1:])
    spline = natural_spline(x, y)
    i = np.searchsorted(x, point) - 1
    value = np.polynomial.polynomial.polyval(point - x[i], spline[i])
    print("\n3.2. Натуральный сплайн, коэффициенты a, b, c, d")
    for left, right, row in zip(x, x[1:], spline):
        print(f"[{left}, {right}]:", row)
    print(f"S({point}) = {value:.10f}")


if __name__ == "__main__":
    main()
