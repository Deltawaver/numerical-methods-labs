import numpy as np

MAX_ITER = 10000


def vector_norm_c(x):
    return np.max(np.abs(x))


def matrix_norm_c(M):
    return np.max(np.sum(np.abs(M), axis=1))


def to_equivalent_form(A, b, n):
    alpha = np.zeros((n, n))
    beta = np.zeros(n)
    for i in range(n):
        beta[i] = b[i] / A[i, i]
        for j in range(n):
            if i != j:
                alpha[i, j] = -A[i, j] / A[i, i]
    return alpha, beta


def has_diagonal_dominance(A, n):
    return all(abs(A[i, i]) > np.sum(np.abs(A[i])) - abs(A[i, i]) for i in range(n))


def a_priori_iterations(alpha_norm, beta, eps):
    k_plus_1 = (np.log10(eps) + np.log10(1 - alpha_norm) - np.log10(vector_norm_c(beta))) \
               / np.log10(alpha_norm)
    return int(np.ceil(k_plus_1 - 1))


def error_estimate(diff_norm, coef_norm, alpha_norm):
    if alpha_norm < 1:
        return coef_norm / (1 - alpha_norm) * diff_norm
    return diff_norm


def simple_iteration(alpha, beta, eps, n):
    alpha_norm = matrix_norm_c(alpha)
    x_prev = beta.copy()                       
    for k in range(1, MAX_ITER + 1):
        x = beta + alpha @ x_prev              
        eps_k = error_estimate(vector_norm_c(x - x_prev), alpha_norm, alpha_norm)
        x_prev = x
        if eps_k <= eps:
            return x, k
    raise RuntimeError("Метод простых итераций не сошёлся")


def seidel(alpha, beta, eps, n):
    alpha_norm = matrix_norm_c(alpha)
    C = np.triu(alpha)                       
    C_norm = matrix_norm_c(C)

    x_prev = beta.copy()                       
    for k in range(1, MAX_ITER + 1):
        x = np.zeros(n)
        for i in range(n):
            s_new = sum(alpha[i, j] * x[j] for j in range(i))         
            s_old = sum(alpha[i, j] * x_prev[j] for j in range(i, n))  
            x[i] = beta[i] + s_new + s_old
        eps_k = error_estimate(vector_norm_c(x - x_prev), C_norm, alpha_norm)
        x_prev = x
        if eps_k <= eps:
            return x, k
    raise RuntimeError("Метод Зейделя не сошёлся")


def seidel_matrix_form(alpha, beta, n):
    B = np.tril(alpha)
    C = np.triu(alpha)
    inv = np.linalg.inv(np.eye(n) - B)
    return inv @ C, inv @ beta


def main():
    data = np.loadtxt("data1.3.txt", ndmin=2)
    n = data.shape[0]
    A = data[:, :n]
    b = data[:, n]
    eps = 0.01
    digits = 4

    alpha, beta = to_equivalent_form(A, b, n)
    alpha_norm = matrix_norm_c(alpha)

    print("Эквивалентная система x = beta + alpha*x")
    print("alpha =\n", np.round(alpha, digits))
    print("beta =", np.round(beta, digits))
    print("||alpha||_c =", round(alpha_norm, digits))
    print("Диагональное преобладание:", has_diagonal_dominance(A, n))
    if alpha_norm < 1:
        print("Достаточное условие сходимости ||alpha|| < 1 выполнено")
        print("Априорная оценка числа итераций: k >=",
              a_priori_iterations(alpha_norm, beta, eps))
    else:
        print("ВНИМАНИЕ: ||alpha|| >= 1, сходимость не гарантирована")

    rho = max(abs(np.linalg.eigvals(alpha)))
    print("Спектральный радиус alpha:", round(rho, digits), "\n")

    x_exact = np.linalg.solve(A, b)
    x_si, k_si = simple_iteration(alpha, beta, eps, n)
    x_se, k_se = seidel(alpha, beta, eps, n)

    print("Точное решение (numpy):", np.round(x_exact, digits), "\n")
    print(f"Метод простых итераций, eps = {eps}:")
    print("  x =", np.round(x_si, digits), " итераций:", k_si)
    print("  реальная погрешность:", vector_norm_c(x_si - x_exact), "\n")
    print(f"Метод Зейделя, eps = {eps}:")
    print("  x =", np.round(x_se, digits), " итераций:", k_se)
    print("  реальная погрешность:", vector_norm_c(x_se - x_exact), "\n")

    M, _ = seidel_matrix_form(alpha, beta, n)
    print("Спектральный радиус матрицы Зейделя (E-B)^-1 C:",
          round(max(abs(np.linalg.eigvals(M))), digits), "\n")

    if k_se < k_si:
        print("Метод Зейделя сошёлся быстрее")
    elif k_si < k_se:
        print("Метод простых итераций сошёлся быстрее")
    else:
        print("Методы потребовали одинакового числа итераций")


if __name__ == "__main__":
    main()