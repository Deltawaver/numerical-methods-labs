import numpy as np


def check_stability(a, b, c, n):
    ok_all = all(abs(b[i]) >= abs(a[i]) + abs(c[i]) for i in range(n))
    ok_strict = any(abs(b[i]) > abs(a[i]) + abs(c[i]) for i in range(n))
    return ok_all and ok_strict


def progonka(a, b, c, d, n):
    P = np.zeros(n)
    Q = np.zeros(n)
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom
    x = np.zeros(n)
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x, P, Q


def main():
    data = np.loadtxt("data1.2.txt", ndmin=2)
    n = data.shape[0]
    A = data[:, :n]
    d = data[:, n]
    digits = 4

    a = np.zeros(n)  
    b = np.zeros(n)  
    c = np.zeros(n) 
    for i in range(n):
        b[i] = A[i, i]
        if i > 0:
            a[i] = A[i, i - 1]
        if i < n - 1:
            c[i] = A[i, i + 1]

    if not check_stability(a, b, c, n):
        print("условие устойчивости (1.8) не выполнено!\n")

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