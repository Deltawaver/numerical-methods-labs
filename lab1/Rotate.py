from pathlib import Path

import sys
import numpy as np

MAX_ITER = 10000


def off_diagonal_norm(A, n):
    """Критерий t(A) = ( sum_{l<m} a_lm^2 )^(1/2) - сумма квадратов внедиагональных элементов."""
    return np.sqrt(np.sum(np.triu(A, k=1) ** 2))


def find_max_off_diagonal(A, n):
    """Индексы (i, j), i < j, максимального по модулю внедиагонального элемента a_ij^(k)."""
    i_max, j_max, a_max = 0, 1, -1.0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > a_max:
                a_max = abs(A[i, j])
                i_max, j_max = i, j
    return i_max, j_max


def rotation_angle(A, i, j):
    """
    Угол вращения из условия a_ij^(k+1) = 0:
        phi = 1/2 * arctg( 2 a_ij / (a_ii - a_jj) ),
    а если a_ii = a_jj, то phi = pi/4 (знак берём по знаку a_ij).
    """
    if np.isclose(A[i, i], A[j, j]):
        return np.sign(A[i, j]) * np.pi / 4
    return 0.5 * np.arctan(2 * A[i, j] / (A[i, i] - A[j, j]))


def rotation_matrix(n, i, j, phi):
    """
    Матрица вращения U^(k):
        u_ii = u_jj = cos(phi),  u_ij = -sin(phi),  u_ji = sin(phi),
        остальные диагональные = 1, остальные внедиагональные = 0.
    """
    U_k = np.eye(n)
    U_k[i, i] = U_k[j, j] = np.cos(phi)
    U_k[i, j] = -np.sin(phi)
    U_k[j, i] = np.sin(phi)
    return U_k


def jacobi_rotations(A, eps, n):
    """
    Метод вращений Якоби.
    На каждой итерации:  A^(k+1) = (U^(k))^T A^(k) U^(k),
    а накопленная матрица  U = U^(0) U^(1) ... U^(k)  содержит собственные векторы по столбцам.
    Возвращает: A^(k) (с собственными значениями на диагонали), U,
    число итераций и историю (t, диагональ, U) по итерациям для анализа погрешности.
    """
    A_k = A.astype(float).copy()
    U = np.eye(n)
    history = [(off_diagonal_norm(A_k, n), np.diag(A_k).copy(), U.copy())]   # k = 0

    k = 0
    while history[-1][0] > eps and k < MAX_ITER:
        i, j = find_max_off_diagonal(A_k, n)
        phi = rotation_angle(A_k, i, j)
        U_k = rotation_matrix(n, i, j, phi)

        A_k = U_k.T @ A_k @ U_k
        A_k[i, j] = A_k[j, i] = 0.0        # обнулённый элемент (убираем ошибку округления)
        U = U @ U_k
        k += 1
        history.append((off_diagonal_norm(A_k, n), np.diag(A_k).copy(), U.copy()))

    return A_k, U, k, history


def error_analysis(A, history, n):
    """
    Таблица погрешностей по итерациям:
      t_k     - критерий остановки,
      err_lam - max |lambda_i^(k) - lambda_i(точное)|  (точное - numpy.linalg.eigvalsh),
      err_ort - ||U^T U - E||  (нарушение ортогональности векторов).
    """
    lam_exact = np.sort(np.linalg.eigvalsh(A))
    rows = []
    for k, (t, diag, U) in enumerate(history):
        err_lam = np.max(np.abs(np.sort(diag) - lam_exact))
        err_ort = np.linalg.norm(U.T @ U - np.eye(n))
        rows.append((k, t, err_lam, err_ort))
    return rows


def save_plot(rows, filename):
    """График погрешности от номера итерации (лог. шкала). Если нет matplotlib - пропускаем."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен - график не построен")
        return
    ks = [r[0] for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(ks, [max(r[1], 1e-18) for r in rows], "o-", label="t(A^(k)) - критерий")
    plt.semilogy(ks, [max(r[2], 1e-18) for r in rows], "s--", label="погрешность собств. значений")
    plt.xlabel("номер итерации k")
    plt.ylabel("погрешность")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print("График сохранён:", filename)


def main():
    # использование: python lab1_4_jacobi_rotations.py [eps] [файл_с_матрицей]
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    filename = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).with_name("data1.4.txt")
    digits = 4

    A = np.loadtxt(filename, ndmin=2)
    n = A.shape[0]
    if A.shape[1] != n or not np.allclose(A, A.T):
        raise ValueError("Метод вращений применим только к симметрическим матрицам")

    A_k, U, k, history = jacobi_rotations(A, eps, n)

    np.set_printoptions(suppress=True)
    print("Точность eps =", eps, " Число итераций:", k)
    print("t(A^(k)) =", history[-1][0], "\n")

    print("Собственные значения:")
    for i in range(n):
        print(f"  lambda{i + 1} = {A_k[i, i]:.{digits}f}")
    print("\nСобственные векторы (столбцы U):")
    for i in range(n):
        print(f"  x{i + 1} =", np.round(U[:, i], digits))

    print("\nПроверка A*x - lambda*x для каждого вектора (норма невязки):")
    for i in range(n):
        print(f"  ||A x{i + 1} - lambda{i + 1} x{i + 1}|| =", np.linalg.norm(A @ U[:, i] - A_k[i, i] * U[:, i]))
    print("\nПроверка ортогональности U^T U:\n", np.round(U.T @ U, digits))
    print("\nПроверка A U и U Lambda:\n", np.round(A @ U, digits), "\n", np.round(U @ np.diag(np.diag(A_k)), digits))

    # --- анализ зависимости погрешности от числа итераций ---
    # считаем заново с очень маленькой eps, чтобы увидеть всю картину сходимости
    _, _, k_full, history_full = jacobi_rotations(A, 1e-12, n)
    rows = error_analysis(A, history_full, n)
    print("\nЗависимость погрешности от числа итераций (eps = 1e-12):")
    print(f"{'k':>3} {'t(A^(k))':>12} {'err lambda':>12} {'err ортог.':>12}")
    for r in rows:
        print(f"{r[0]:>3} {r[1]:>12.3e} {r[2]:>12.3e} {r[3]:>12.3e}")
    save_plot(rows, "jacobi_error.png")


if __name__ == "__main__":
    main()