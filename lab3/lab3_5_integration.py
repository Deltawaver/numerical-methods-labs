import math
import sys
from pathlib import Path

import numpy as np


def quadrature(f, a, b, h):
    n = round((b - a) / h)
    if n < 2 or n % 2 or not math.isclose(n * h, b - a):
        raise ValueError("Для Симпсона нужно чётное число полных шагов")
    nodes = np.linspace(a, b, n + 1)
    values = f(nodes)
    endpoints = values[0] + values[-1]
    return np.array([
        h * sum(f((nodes[:-1] + nodes[1:]) / 2)),
        h * (endpoints / 2 + sum(values[1:-1])),
        h / 3 * (endpoints + 4 * sum(values[1::2]) + 2 * sum(values[2:-1:2])),
    ])


def main():
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("data3.5.txt")
    rows = filename.read_text(encoding="utf-8-sig").splitlines()
    a, b, h1, h2 = map(float, rows[0].split())
    print(f"3.5. Интеграл 1/(x² + 4) от {a:g} до {b:g}")
    def integrand(x):
        return 1 / (x**2 + 4)

    coarse = quadrature(integrand, a, b, h1)
    fine = quadrature(integrand, a, b, h2)
    correction = (fine - coarse) / ((h1 / h2)**np.array([2, 2, 4]) - 1)
    for name, q1, q2, delta in zip(
        ("Прямоугольники", "Трапеции", "Симпсон"), coarse, fine, correction,
    ):
        print(f"{name}: h={h1}: {q1:.10f}, h={h2}: {q2:.10f}, "
              f"оценка ошибки {abs(delta):.10g}, уточнение {q2 + delta:.10f}")
    print(f"Точное значение: {(math.atan(b / 2) - math.atan(a / 2)) / 2:.10f}")


if __name__ == "__main__":
    main()
