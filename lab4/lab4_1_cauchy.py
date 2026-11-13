from pathlib import Path
import sys
import math
import numpy as np


def grid(a, b, h):
    if not math.isfinite(h) or h <= 0:
        raise ValueError("Шаг должен быть положительным")
    n = round((b - a) / h)
    if n < 4 or not math.isclose(n * h, b - a):
        raise ValueError("Шаг должен делить отрезок, минимум четыре шага")
    return np.linspace(a, b, n + 1)


def rhs(x, state):
    y, z = state
    return np.array([z, 4*x*z - (4*x*x - 3)*y + np.exp(x*x)])


def rk4_step(f, x, state, h):
    k1 = f(x, state)
    k2 = f(x + h/2, state + h*k1/2)
    k3 = f(x + h/2, state + h*k2/2)
    k4 = f(x + h, state + h*k3)
    return state + h*(k1 + 2*k2 + 2*k3 + k4)/6


def solve(x, initial, method):
    h = x[1] - x[0]
    states = np.zeros((len(x), 2))
    states[0] = initial
    for i in range(len(x)-1):
        if method == "Эйлер":
            states[i+1] = states[i] + h*rhs(x[i], states[i])
        elif method == "Рунге-Кутта" or i < 3:
            states[i+1] = rk4_step(rhs, x[i], states[i], h)
        else:
            values = [rhs(x[i-j], states[i-j]) for j in range(4)]
            states[i+1] = states[i] + h/24*(
                55*values[0] - 59*values[1] + 37*values[2] - 9*values[3])
    return states[:, 0]


def main():
    filename = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name("data4.1.txt")
    a, b, h, y0, z0 = np.loadtxt(filename)
    if len(sys.argv)>2:
        h = float(sys.argv[2])
    x, fine_x = grid(a,b,h), grid(a,b,h/2)
    exact = (np.exp(x)+np.exp(-x)-1)*np.exp(x*x)
    print("4.1. Вариант 8, задача Коши")
    for method, order in [("Эйлер",1),("Рунге-Кутта",4),("Адамс",4)]:
        y = solve(x, [y0,z0], method)
        fine = solve(fine_x, [y0,z0], method)[::2]
        rr = np.abs(fine-y)/(2**order-1)
        print(f"\n{method}: x, y, точное, ошибка, оценка Рунге–Ромберга мелкой сетки")
        for row in zip(x,y,exact,np.abs(y-exact),rr):
            print(" ".join(f"{v: .10f}" for v in row))
        print(f"Максимальная ошибка: {max(abs(y-exact)):.6g}; РР: {max(rr):.6g}")


if __name__ == "__main__":
    main()
