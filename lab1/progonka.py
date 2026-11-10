from pathlib import Path

import numpy as np


def check_stability(a, b, c, n):
    """
    Условие устойчивости (1.8): |b_i| >= |a_i| + |c_i| для всех i,
    причём хотя бы в одной строке неравенство строгое.
    """
    ok_all = all(abs(b[i]) >= abs(a[i]) + abs(c[i]) for i in range(n))
    ok_strict = any(abs(b[i]) > abs(a[i]) + abs(c[i]) for i in range(n))
    return ok_all and ok_strict


def progonka(a, b, c, d, n):
    """
    Правая прогонка, формулы (1.4)-(1.7).
    Система:  a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i,  a_1 = 0, c_n = 0.
    Возвращает x, прогоночные коэффициенты P, Q.
    """
    P = np.zeros(n)
    Q = np.zeros(n)

    # Прямой ход
    # (1.5): P_1 = -c_1 / b_1,  Q_1 = d_1 / b_1
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    # (1.4): P_i = -c_i / (b_i + a_i P_{i-1}),
    #        Q_i = (d_i - a_i Q_{i-1}) / (b_i + a_i P_{i-1})
    # (1.6): для i = n получится P_n = 0, т.к. c_n = 0
    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom

    # Обратный ход (1.7): x_n = Q_n,  x_i = P_i x_{i+1} + Q_i
    x = np.zeros(n)
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x, P, Q


def main():
    # файл: n строк; в каждой полная строка матрицы (n чисел) и правая часть d_i
    data = np.loadtxt(Path(__file__).with_name("data1.2.txt"), ndmin=2)
    n = data.shape[0]
    A = data[:, :n]
    d = data[:, n]
    digits = 4

    # выделяем три диагонали
    a = np.zeros(n)   # нижняя:  a_i = A[i, i-1],  a_1 = 0
    b = np.zeros(n)   # главная: b_i = A[i, i]
    c = np.zeros(n)   # верхняя: c_i = A[i, i+1],  c_n = 0
    for i in range(n):
        b[i] = A[i, i]
        if i > 0:
            a[i] = A[i, i - 1]
        if i < n - 1:
            c[i] = A[i, i + 1]

    if not check_stability(a, b, c, n):
        print("Внимание: условие устойчивости (1.8) не выполнено!\n")

    x, P, Q = progonka(a, b, c, d, n)

    print("Прогоночные коэффициенты:")
    for i in range(n):
        print(f"  P{i + 1} = {P[i]:.{digits}f},  Q{i + 1} = {Q[i]:.{digits}f}")

    print("\nРешение x:")
    for i in range(n):
        print(f"  x{i + 1} = {x[i]:.{digits}f}")

    print("\nНевязка ||Ax - d||:", np.linalg.norm(A @ x - d))
    print("Проверка через numpy.linalg.solve:", np.round(np.linalg.solve(A, d), digits))


if __name__ == "__main__":
    main()