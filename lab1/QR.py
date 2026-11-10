from pathlib import Path

import sys
import numpy as np

MAX_ITER = 10000
ZERO = 1e-14


def householder_matrix(v):
    """Матрица Хаусхолдера (1.29):  H = E - 2 v v^T / (v^T v)."""
    v = v.reshape(-1, 1)
    return np.eye(len(v)) - 2 * (v @ v.T) / (v.T @ v)


def qr_decomposition(A, n):
    """
    QR-разложение с помощью преобразований Хаусхолдера: A = Q R.
    На k-м шаге обнуляются поддиагональные элементы k-го столбца матрицы A^(k-1):
        b = (a_kk, a_{k+1,k}, ..., a_nk)^T,
        v = b + sign(b_1) * ||b||_2 * e_1            (1.30)
    Тогда H_k * b = (-sign(b_1)*||b||, 0, ..., 0)^T.
    Итог:  R = H_{n-1} ... H_1 A,   Q = H_1 H_2 ... H_{n-1}  (H симметрична и ортогональна).
    """
    R = A.astype(float).copy()             # A^(k) - матрица после k шагов
    Q = np.eye(n)

    for k in range(n - 1):
        b = R[k:, k]                       # часть столбца с диагонали и ниже
        b_norm = np.linalg.norm(b)
        if np.linalg.norm(b[1:]) < ZERO:   # поддиагональные элементы уже нули
            continue

        sign = 1.0 if b[0] >= 0 else -1.0  # sign(0) считаем равным 1
        v = b.copy()
        v[0] += sign * b_norm              # v = b + sign(b_1) ||b|| e_1

        H_k = np.eye(n)                    # H_k = diag(E_k, H~), H~ строится по v
        H_k[k:, k:] = householder_matrix(v)

        R = H_k @ R
        Q = Q @ H_k

    R = np.triu(R)                         # убираем ошибки округления под диагональю
    return Q, R


def extract_eigenvalues(A_k, n, eps, lam_prev):
    """
    Анализ матрицы A^(k) (критерии окончания из методички).

    * Вещественное собственное значение: поддиагональная часть столбца j стремится к нулю,
      критерий  ( sum_{m>j} a_mj^2 )^(1/2) <= eps,  тогда lambda_j = a_jj.
    * Комплексно-сопряжённая пара: диагональный блок 2x2 (столбцы j, j+1).
      Элементы блока от итерации к итерации меняются, а корни уравнения
         lambda^2 - (a_jj + a_{j+1,j+1}) lambda + (a_jj a_{j+1,j+1} - a_{j,j+1} a_{j+1,j}) = 0
      сходятся, критерий  |lambda^(k) - lambda^(k-1)| <= eps.

    Возвращает массив собственных значений и признак, что сошлись все.
    """
    lam = np.zeros(n, dtype=complex)
    converged = True
    j = 0
    while j < n:
        # столбец с малой поддиагональной частью -> вещественное собственное значение
        if np.linalg.norm(A_k[j + 1:, j]) <= eps:
            lam[j] = A_k[j, j]
            j += 1
            continue

        # иначе пробуем увидеть блок 2x2 с комплексными корнями
        is_block = False
        if j + 1 < n:
            isolated = (np.linalg.norm(A_k[j + 2:, j]) <= eps and
                        np.linalg.norm(A_k[j + 2:, j + 1]) <= eps)
            if isolated:
                a, b = A_k[j, j], A_k[j, j + 1]
                c, d = A_k[j + 1, j], A_k[j + 1, j + 1]
                trace, det = a + d, a * d - b * c
                disc = trace ** 2 - 4 * det
                if disc < 0:
                    lam[j] = complex(trace / 2, np.sqrt(-disc) / 2)
                    lam[j + 1] = complex(trace / 2, -np.sqrt(-disc) / 2)
                    if (lam_prev is None or
                            abs(lam[j] - lam_prev[j]) > eps or
                            abs(lam[j + 1] - lam_prev[j + 1]) > eps):
                        converged = False
                    is_block = True

        if is_block:
            j += 2
        else:
            lam[j] = A_k[j, j]          # столбец ещё не сошёлся
            converged = False
            j += 1

    return lam, converged


def qr_algorithm(A, eps, n):
    """
    QR-алгоритм:  A^(0) = A,   A^(k) = Q^(k) R^(k),   A^(k+1) = R^(k) Q^(k).
    Все A^(k) ортогонально подобны A (A^(k+1) = Q^T A^(k) Q).
    """
    A_k = A.astype(float).copy()
    lam_prev = None
    for k in range(1, MAX_ITER + 1):
        Q_k, R_k = qr_decomposition(A_k, n)
        A_k = R_k @ Q_k
        lam, converged = extract_eigenvalues(A_k, n, eps, lam_prev)
        if converged:
            return lam, A_k, k
        lam_prev = lam
    raise RuntimeError("QR-алгоритм не сошёлся за MAX_ITER итераций")


def format_complex(z, digits):
    if abs(z.imag) < 1e-12:
        return f"{z.real:.{digits}f}"
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.{digits}f} {sign} {abs(z.imag):.{digits}f}i"


def main():
    # использование: python lab1_5_qr.py [eps] [файл_с_матрицей]
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    filename = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).with_name("data1.5.txt")
    digits = 4

    A = np.loadtxt(filename, ndmin=2)
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Матрица должна быть квадратной")

    np.set_printoptions(suppress=True)

    # --- часть 1: QR-разложение ---
    Q, R = qr_decomposition(A, n)
    print("Матрица A:\n", np.round(A, digits), "\n")
    print("Матрица Q:\n", np.round(Q, digits), "\n")
    print("Матрица R:\n", np.round(R, digits), "\n")
    print("Проверка Q*R:\n", np.round(Q @ R, digits), "\n")
    print("Проверка ортогональности Q^T Q:\n", np.round(Q.T @ Q, digits), "\n")

    # --- часть 2: QR-алгоритм ---
    lam, A_k, k = qr_algorithm(A, eps, n)
    print(f"QR-алгоритм, eps = {eps}, итераций: {k}\n")
    print("Итоговая матрица A^(k):\n", np.round(A_k, digits), "\n")
    print("Собственные значения:")
    for i in range(n):
        print(f"  lambda{i + 1} = {format_complex(lam[i], digits)}")

    # --- сравнение с numpy ---
    lam_np = np.linalg.eigvals(A)
    key = lambda z: (round(z.real, 6), round(z.imag, 6))
    err = np.max(np.abs(np.array(sorted(lam, key=key)) - np.array(sorted(lam_np, key=key))))
    print("\nСобственные значения numpy.linalg.eigvals:",
          ", ".join(format_complex(z, digits) for z in sorted(lam_np, key=key)))
    print("Максимальное отклонение от numpy:", err)


if __name__ == "__main__":
    main()