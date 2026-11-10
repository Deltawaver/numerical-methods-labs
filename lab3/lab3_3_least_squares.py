import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab1.LU import gauss_lu, solve_lu


def least_squares(x, y, degree):
    design = np.vander(x, degree + 1, increasing=True)
    normal = design.T @ design
    p, l, u, _ = gauss_lu(normal, degree + 1)
    coefficients = solve_lu(p, l, u, design.T @ y, degree + 1)
    residual = y - design @ coefficients
    return coefficients, residual @ residual


def main():
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("data3.3.txt")
    rows = filename.read_text(encoding="utf-8-sig").splitlines()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n3.3. МНК")
    x, y = (np.fromstring(row, sep=" ") for row in rows)
    grid = np.linspace(x[0], x[-1], 300)
    fig, ax = plt.subplots()
    ax.plot(x, y, "o--", label="Табличная функция")
    for degree in (1, 2):
        coefficients, error = least_squares(x, y, degree)
        print(f"Степень {degree}: {coefficients}, сумма квадратов ошибок {error:.10g}")
        ax.plot(grid, np.polynomial.polynomial.polyval(grid, coefficients),
                label=f"МНК, степень {degree}")
    ax.set(xlabel="x", ylabel="y", title="Приближение методом наименьших квадратов")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("lab3_3.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
