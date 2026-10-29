import sys
import math
import numpy as np

MAX_ITER = 10000
HEADERS = {"equations", "jacobian", "phi", "x0", "region"}

SAFE_NAMES = {name: getattr(math, name) for name in
              ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
               "exp", "log", "log10", "sqrt", "pi", "e")}
SAFE_NAMES["ln"] = math.log
SAFE_NAMES["abs"] = abs


# ---------------------------------------------------------------- чтение входных данных

def compile_expression(text, variables):
    """Строка вида 'x1**2 + x2**2 - 4' -> функция от (x1, ..., xn)."""
    code = compile(text.replace("^", "**"), "<expression>", "eval")

    def func(*args):
        env = dict(SAFE_NAMES)
        env.update(zip(variables, args))
        return eval(code, {"__builtins__": {}}, env)
    return func


def read_sections(filename):
    sections, current = {}, None
    with open(filename, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.lower() in HEADERS:
                current = line.lower()
                sections[current] = []
            elif current is not None:
                sections[current].append(line)
    return sections


def make_vector_function(expressions, n):
    """Список строк -> векторная функция F(x), x - массив длины n, переменные x1..xn."""
    variables = [f"x{i + 1}" for i in range(n)]
    funcs = [compile_expression(text, variables) for text in expressions]
    return lambda x: np.array([fn(*x) for fn in funcs], dtype=float)


def make_matrix_function(rows, n):
    """n строк по n выражений через ';' -> матричная функция J(x) (аналитический якобиан)."""
    variables = [f"x{i + 1}" for i in range(n)]
    funcs = [[compile_expression(text.strip(), variables) for text in row.split(";")] for row in rows]
    return lambda x: np.array([[fn(*x) for fn in row] for row in funcs], dtype=float)


# ---------------------------------------------------------------- линейная алгебра (свои реализации)

def vector_norm_c(x):
    return np.max(np.abs(x))


def matrix_norm_c(M):
    return np.max(np.sum(np.abs(M), axis=1))


def gauss_solve(A, b):
    """Решение A x = b методом Гаусса с выбором ведущего элемента (как в лабе 1.1)."""
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    for k in range(n):
        lead = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[lead, k]) < 1e-14:
            raise ValueError("Матрица Якоби вырождена")
        if lead != k:
            M[[k, lead]] = M[[lead, k]]
        for i in range(k + 1, n):
            mu = M[i, k] / M[k, k]
            M[i, k:] -= mu * M[k, k:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x


def inverse_matrix(A):
    """Обратная матрица: решаем A x = e_j для каждого столбца единичной матрицы."""
    n = len(A)
    E = np.eye(n)
    X = np.zeros((n, n))
    for j in range(n):
        X[:, j] = gauss_solve(A, E[:, j])
    return X


def numeric_jacobian(F, x, h=1e-6):
    """Матрица Якоби центральными разностями: J_ij = dF_i/dx_j."""
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        J[:, j] = (F(x_plus) - F(x_minus)) / (2 * h)
    return J


# ---------------------------------------------------------------- методы

def simple_iteration(Phi, x0, eps):
    """
    Метод простой итерации: x^(k) = Phi(x^(k-1)).
    q = ||Phi'(x)|| (матрица Якоби Phi) оценивается в точках итераций (берём максимум).
    Если q < 1:  ||x^(k) - x*|| <= q/(1-q) * ||x^(k) - x^(k-1)||, иначе смотрим на ||x^(k) - x^(k-1)||.
    """
    x_prev = x0.copy()
    q = matrix_norm_c(numeric_jacobian(Phi, x_prev))
    history = []                                   # (k, x_k, ||x_k - x_{k-1}||, оценка, q)
    for k in range(1, MAX_ITER + 1):
        x = Phi(x_prev)
        if not np.all(np.isfinite(x)):
            raise RuntimeError("Метод простой итерации расходится")
        q = max(q, matrix_norm_c(numeric_jacobian(Phi, x)))
        delta = vector_norm_c(x - x_prev)
        estimate = q / (1 - q) * delta if q < 1 else delta
        history.append((k, x.copy(), delta, estimate, q))
        x_prev = x
        if estimate <= eps:
            return x, k, history
    raise RuntimeError("Метод простой итерации не сошёлся")


def newton(F, J, x0, eps):
    """
    Метод Ньютона: J(x^(k-1)) * dx = -F(x^(k-1)),  x^(k) = x^(k-1) + dx.
    Остановка по ||x^(k) - x^(k-1)|| <= eps.
    """
    x_prev = x0.copy()
    history = []
    for k in range(1, MAX_ITER + 1):
        dx = gauss_solve(J(x_prev), -F(x_prev))
        x = x_prev + dx
        delta = vector_norm_c(dx)
        history.append((k, x.copy(), delta, delta, 0.0))
        x_prev = x
        if delta <= eps:
            return x, k, history
    raise RuntimeError("Метод Ньютона не сошёлся")


def reference_solution(F, J, x0):
    """'Точное' решение: Ньютон до предела машинной точности (для анализа погрешности)."""
    x = x0.copy()
    for _ in range(100):
        dx = gauss_solve(J(x), -F(x))
        x = x + dx
        if vector_norm_c(dx) <= 1e-15:
            break
    return x


# ---------------------------------------------------------------- вывод и графики

def print_table(title, history, x_ref):
    print(title)
    print(f"{'k':>3} {'||x_k-x_(k-1)||':>16} {'оценка погр.':>14} {'||x_k-x*||':>12}")
    for k, x, delta, estimate, _ in history:
        print(f"{k:>3} {delta:>16.3e} {estimate:>14.3e} {vector_norm_c(x - x_ref):>12.3e}")
    print()


def save_graph(F, region, x0, solution, filename):
    """Для n = 2: линии F1 = 0 и F2 = 0 на плоскости - пересечения и есть решения."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен - графики не построены")
        return
    x_min, x_max, y_min, y_max = region
    steps = 150
    xs = [x_min + (x_max - x_min) * i / steps for i in range(steps + 1)]
    ys = [y_min + (y_max - y_min) * j / steps for j in range(steps + 1)]
    Z1 = np.zeros((len(ys), len(xs)))
    Z2 = np.zeros((len(ys), len(xs)))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            values = F(np.array([x, y]))
            Z1[j, i], Z2[j, i] = values[0], values[1]
    plt.figure(figsize=(6, 6))
    plt.contour(xs, ys, Z1, levels=[0], colors="tab:blue")
    plt.contour(xs, ys, Z2, levels=[0], colors="tab:orange")
    plt.plot([], [], color="tab:blue", label="F1(x1, x2) = 0")
    plt.plot([], [], color="tab:orange", label="F2(x1, x2) = 0")
    plt.plot([x0[0]], [x0[1]], "go", label="начальное приближение")
    plt.plot([solution[0]], [solution[1]], "r*", markersize=12, label="решение")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print("График сохранён:", filename)


def save_error_plot(hist_si, hist_nw, x_ref, filename):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(7, 4.5))
    for history, label, style in ((hist_si, "простая итерация", "o-"), (hist_nw, "Ньютон", "s--")):
        plt.semilogy([h[0] for h in history],
                     [max(vector_norm_c(h[1] - x_ref), 1e-18) for h in history], style, label=label)
    plt.xlabel("номер итерации k")
    plt.ylabel("||x_k - x*||")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print("График сохранён:", filename)


def main():
    # использование: python lab2_2_systems.py [eps] [файл]
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
    filename = sys.argv[2] if len(sys.argv) > 2 else "data2.2.txt"
    digits = 10

    sections = read_sections(filename)
    equations = sections["equations"]
    n = len(equations)
    x0 = np.array(list(map(float, sections["x0"][0].split())))
    if len(x0) != n:
        raise ValueError("Число компонент x0 должно совпадать с числом уравнений")

    F = make_vector_function(equations, n)

    jacobian_lines = sections.get("jacobian", ["auto"])
    if jacobian_lines[0].lower() == "auto":
        J = lambda x: numeric_jacobian(F, x)
    else:
        J = make_matrix_function(jacobian_lines, n)

    phi_lines = sections.get("phi", ["auto"])
    if phi_lines[0].lower() == "auto":
        # Phi(x) = x - Lambda * F(x),  Lambda = J(x0)^(-1)  =>  Phi'(x) = E - Lambda * J(x)
        Lambda = inverse_matrix(J(x0))
        Phi = lambda x: x - Lambda @ F(x)
    else:
        Phi = make_vector_function(phi_lines, n)

    print(f"Точность eps = {eps},  n = {n},  x0 = {x0}\n")

    x_ref = reference_solution(F, J, x0)
    x_si, k_si, hist_si = simple_iteration(Phi, x0, eps)
    x_nw, k_nw, hist_nw = newton(F, J, x0, eps)

    print("Простая итерация: ||Phi'(x0)|| =", round(matrix_norm_c(numeric_jacobian(Phi, x0)), 6),
          ", q (макс. по итерациям) =", round(hist_si[-1][4], 6))
    print("  x =", np.round(x_si, digits), " итераций:", k_si,
          " ||x - x*|| =", f"{vector_norm_c(x_si - x_ref):.3e}")
    print("Метод Ньютона:")
    print("  x =", np.round(x_nw, digits), " итераций:", k_nw,
          " ||x - x*|| =", f"{vector_norm_c(x_nw - x_ref):.3e}")
    print("Эталон (Ньютон до 1e-15): x* =", np.round(x_ref, 12))
    print("Невязка ||F(x*)|| =", f"{vector_norm_c(F(x_ref)):.3e}")
    print("Все компоненты положительны:", bool(np.all(x_nw > 0)), "\n")

    # --- анализ зависимости погрешности от числа итераций (до предела точности)
    _, _, full_si = simple_iteration(Phi, x0, 1e-14)
    _, _, full_nw = newton(F, J, x0, 1e-14)
    print_table("Простая итерация: погрешность по итерациям", full_si, x_ref)
    print_table("Метод Ньютона: погрешность по итерациям", full_nw, x_ref)

    if n == 2 and "region" in sections:
        region = list(map(float, sections["region"][0].split()))
        save_graph(F, region, x0, x_ref, "lab2_2_graph.png")
    save_error_plot(full_si, full_nw, x_ref, "lab2_2_error.png")


if __name__ == "__main__":
    main()
