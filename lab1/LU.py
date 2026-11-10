from pathlib import Path

import numpy as np

EPS_ZERO = 1e-12   # ниже этого порога считаем ведущий элемент нулём


def gauss_lu(A, n):
    """
    Прямой ход метода Гаусса -> LU-разложение (Замечание 5 методички).
    Возвращает P, L, U, p  такие, что  P*A = L*U,
    p - число перестановок строк (нужно для определителя, Замечание 3).
    """
    A_k = A.astype(float).copy()   # A^(k) - матрица на k-м шаге
    L = np.eye(n)                  # в поддиагонали будут храниться mu_ik
    P = np.eye(n)                  # матрица перестановок
    p = 0

    for k in range(n - 1):         # шаги 1..n-1 прямого хода
        # выбор ведущего элемента (по модулю) в k-м столбце ниже диагонали
        lead = k + np.argmax(np.abs(A_k[k:, k]))
        if abs(A_k[lead, k]) < EPS_ZERO:
            raise ValueError("Матрица вырождена (det A = 0)")

        if lead != k:              # переставляем строки
            A_k[[k, lead]] = A_k[[lead, k]]
            P[[k, lead]] = P[[lead, k]]
            L[[k, lead], :k] = L[[lead, k], :k]   # уже найденные mu тоже переставляются
            p += 1

        # mu_ik = a_ik^(k-1) / a_kk^(k-1),  i = k+1..n
        mu = A_k[k + 1:, k] / A_k[k, k]
        L[k + 1:, k] = mu

        # a_ij^(k) = a_ij^(k-1) - mu_ik * a_kj^(k-1)
        A_k[k + 1:, :] -= np.outer(mu, A_k[k, :])

    if abs(A_k[n - 1, n - 1]) < EPS_ZERO:
        raise ValueError("Матрица вырождена (det A = 0)")

    U = np.triu(A_k)               # A^(n-1) - верхняя треугольная
    return P, L, U, p


def solve_lu(P, L, U, b, n):
    """
    Решение A x = b через LU:  L z = P b,  U x = z.
    """
    Pb = P @ b

    # 1-й этап: z_i = b_i - sum_{j<i} l_ij z_j
    z = np.zeros(n)
    for i in range(n):
        z[i] = Pb[i] - sum(L[i, j] * z[j] for j in range(i))

    # 2-й этап (обратный ход): x_i = (z_i - sum_{j>i} u_ij x_j) / u_ii
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x


def inverse_matrix(P, L, U, n):
    """
    Замечание 4: A X = E.  Решаем n СЛАУ с правыми частями - столбцами E,
    LU-разложение при этом строится только один раз.
    """
    E = np.eye(n)
    X = np.zeros((n, n))           # X = A^(-1)
    for j in range(n):
        X[:, j] = solve_lu(P, L, U, E[:, j], n)
    return X


def determinant(U, p, n):
    """det A = (-1)^p * a11 * a22^(1) * ... * ann^(n-1)  (Замечание 3)."""
    return (-1) ** p * np.prod(np.diag(U))


def main():
    # файл: n строк, в каждой n коэффициентов матрицы и правая часть b
    data = np.loadtxt(Path(__file__).with_name("data1.1.txt"), ndmin=2)
    n = data.shape[0]
    A = data[:, :n]
    b = data[:, n]
    digits = 4

    P, L, U, p = gauss_lu(A, n)
    x = solve_lu(P, L, U, b, n)
    A_inv = inverse_matrix(P, L, U, n)
    det_A = determinant(U, p, n)

    np.set_printoptions(suppress=True)
    print("Матрица L:\n", np.round(L, digits), "\n")
    print("Матрица U:\n", np.round(U, digits), "\n")
    print("Проверка P*A:\n", np.round(P @ A, digits))
    print("Проверка L*U:\n", np.round(L @ U, digits), "\n")
    print("Решение x:\n", np.round(x, digits), "\n")
    print("Невязка ||Ax - b||:", np.linalg.norm(A @ x - b), "\n")
    print("Определитель det A =", round(det_A, digits),
          "(numpy:", round(np.linalg.det(A), digits), ")\n")
    print("Обратная матрица A^-1:\n", np.round(A_inv, digits), "\n")
    print("Проверка A * A^-1:\n", np.round(A @ A_inv, digits))


if __name__ == "__main__":
    main()