import numpy as np

EPS_ZERO = 1e-12  


def gauss_lu(A, n):
    A_k = A.astype(float).copy()   
    L = np.eye(n)                 
    P = np.eye(n)                  
    p = 0

    for k in range(n - 1):        
        lead = k + np.argmax(np.abs(A_k[k:, k]))
        if abs(A_k[lead, k]) < EPS_ZERO:
            raise ValueError("Матрица вырождена (det A = 0)")
        if lead != k:              
            A_k[[k, lead]] = A_k[[lead, k]]
            P[[k, lead]] = P[[lead, k]]
            L[[k, lead], :k] = L[[lead, k], :k]  
            p += 1
        mu = A_k[k + 1:, k] / A_k[k, k]
        L[k + 1:, k] = mu
        A_k[k + 1:, :] -= np.outer(mu, A_k[k, :])

    if abs(A_k[n - 1, n - 1]) < EPS_ZERO:
        raise ValueError("Матрица вырождена (det A = 0)")

    U = np.triu(A_k)              
    return P, L, U, p


def solve_lu(P, L, U, b, n):
    Pb = P @ b
    z = np.zeros(n)
    for i in range(n):
        z[i] = Pb[i] - sum(L[i, j] * z[j] for j in range(i))

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x


def inverse_matrix(P, L, U, n):
    E = np.eye(n)
    X = np.zeros((n, n))          
    for j in range(n):
        X[:, j] = solve_lu(P, L, U, E[:, j], n)
    return X


def determinant(U, p, n):
    return (-1) ** p * np.prod(np.diag(U))


def main():
    data = np.loadtxt("data1.1.txt", ndmin=2)
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
    print("Определитель det A =", round(det_A, digits),
          "(numpy:", round(np.linalg.det(A), digits), ")\n")
    print("Обратная матрица A^-1:\n", np.round(A_inv, digits), "\n")
    print("Проверка A * A^-1:\n", np.round(A @ A_inv, digits))


if __name__ == "__main__":
    main()