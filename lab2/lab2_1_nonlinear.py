from pathlib import Path

import sys
import math

MAX_ITER = 10000
SAMPLES = 2000                      # число точек для оценки min/max на отрезке [a, b]
HEADERS = {"f", "df", "phi", "interval", "x0"}

SAFE_NAMES = {name: getattr(math, name) for name in
              ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
               "exp", "log", "log10", "sqrt", "pi", "e")}
SAFE_NAMES["ln"] = math.log
SAFE_NAMES["abs"] = abs


# ---------------------------------------------------------------- чтение входных данных

def compile_expression(text, variables):
    """Превращает строку вида 'exp(x) - 2*x' в функцию (знак ^ понимается как степень)."""
    code = compile(text.replace("^", "**"), "<expression>", "eval")

    def func(*args):
        env = dict(SAFE_NAMES)
        env.update(zip(variables, args))
        return eval(code, {"__builtins__": {}}, env)
    return func


def read_sections(filename):
    """Файл разбит на секции: строка-заголовок (f, df, phi, interval, x0) и данные под ней."""
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


# ---------------------------------------------------------------- вспомогательные численные функции

def derivative(f, x, h=1e-6):
    """f'(x) центральной разностью (используется, если df = auto, и для оценок q)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def second_derivative(f, x, h=1e-4):
    """f''(x) центральной разностью."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2


def sample_range(func, a, b):
    """Минимум и максимум функции на [a, b] по равномерной сетке."""
    values = [func(a + (b - a) * i / SAMPLES) for i in range(SAMPLES + 1)]
    return min(values), max(values)


def find_sign_changes(f, a, b, steps=1000):
    """Отрезки, на концах которых f имеет разные знаки (подсказка для графического выбора x0)."""
    result = []
    x_left, f_left = a, f(a)
    for i in range(1, steps + 1):
        x_right = a + (b - a) * i / steps
        f_right = f(x_right)
        if f_left * f_right < 0:
            result.append((x_left, x_right))
        x_left, f_left = x_right, f_right
    return result


def build_auto_phi(f, df, a, b):
    """
    Если phi не задана: x = phi(x) = x - s * f(x) / M,  M = max |f'(x)|, s = sign f'(x) на [a, b].
    Тогда 0 <= phi'(x) = 1 - s f'(x) / M <= 1 - m/M < 1.
    """
    low, high = sample_range(df, a, b)
    if low * high <= 0:
        raise ValueError("f'(x) меняет знак на [a, b] - автоматическое построение phi невозможно")
    M = max(abs(low), abs(high))
    s = 1.0 if low > 0 else -1.0
    return lambda x: x - s * f(x) / M


# ---------------------------------------------------------------- методы

def simple_iteration(phi, x0, eps, q):
    """
    Метод простой итерации: x^(k) = phi(x^(k-1)).
    Оценка погрешности: |x^(k) - x*| <= q/(1-q) * |x^(k) - x^(k-1)|   (если q < 1).
    """
    x_prev = x0
    history = []                                   # (k, x_k, |x_k - x_{k-1}|, оценка погрешности)
    for k in range(1, MAX_ITER + 1):
        x = phi(x_prev)
        if not math.isfinite(x):
            raise RuntimeError("Метод простой итерации расходится")
        delta = abs(x - x_prev)
        estimate = q / (1 - q) * delta if q < 1 else delta
        history.append((k, x, delta, estimate))
        x_prev = x
        if estimate <= eps:
            return x, k, history
    raise RuntimeError("Метод простой итерации не сошёлся")


def newton(f, df, x0, eps):
    """
    Метод Ньютона: x^(k) = x^(k-1) - f(x^(k-1)) / f'(x^(k-1)).
    Остановка по |x^(k) - x^(k-1)| <= eps.
    """
    x_prev = x0
    history = []
    for k in range(1, MAX_ITER + 1):
        slope = df(x_prev)
        if abs(slope) < 1e-14:
            raise RuntimeError("f'(x) = 0 - метод Ньютона неприменим")
        x = x_prev - f(x_prev) / slope
        delta = abs(x - x_prev)
        history.append((k, x, delta, delta))
        x_prev = x
        if delta <= eps:
            return x, k, history
    raise RuntimeError("Метод Ньютона не сошёлся")


def reference_root(f, df, x0):
    """'Точное' значение корня: Ньютон до предела машинной точности (для анализа погрешности)."""
    x = x0
    for _ in range(100):
        x_new = x - f(x) / df(x)
        if abs(x_new - x) <= 1e-15:
            return x_new
        x = x_new
    return x


# ---------------------------------------------------------------- вывод и графики

def print_table(title, history, x_ref):
    print(title)
    print(f"{'k':>3} {'x_k':>16} {'|x_k-x_(k-1)|':>15} {'оценка погр.':>14} {'|x_k-x*|':>12}")
    for k, x, delta, estimate in history:
        print(f"{k:>3} {x:>16.12f} {delta:>15.3e} {estimate:>14.3e} {abs(x - x_ref):>12.3e}")
    print()


def save_graph(f, a, b, x0, root, filename):
    """График f(x) на [a, b] - по нему выбирается начальное приближение."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен - графики не построены")
        return
    xs = [a + (b - a) * i / 500 for i in range(501)]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, [f(x) for x in xs], label="f(x)")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.plot([x0], [f(x0)], "go", label="начальное приближение x0")
    plt.plot([root], [0], "r*", markersize=12, label="корень")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print("График сохранён:", filename)


def save_error_plot(hist_si, hist_nw, x_ref, filename):
    """Зависимость истинной погрешности |x_k - x*| от номера итерации (лог. шкала)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(7, 4.5))
    for history, label, style in ((hist_si, "простая итерация", "o-"), (hist_nw, "Ньютон", "s--")):
        plt.semilogy([h[0] for h in history], [max(abs(h[1] - x_ref), 1e-18) for h in history],
                     style, label=label)
    plt.xlabel("номер итерации k")
    plt.ylabel("|x_k - x*|")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print("График сохранён:", filename)


def main():
    # использование: python lab2_1_nonlinear.py [eps] [файл]
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
    filename = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).with_name("data2.1.txt")

    sections = read_sections(filename)
    f = compile_expression(sections["f"][0], ["x"])
    a, b = map(float, sections["interval"][0].split())
    x0 = float(sections["x0"][0])

    df_text = sections.get("df", ["auto"])[0]
    df = (lambda x: derivative(f, x)) if df_text.lower() == "auto" else compile_expression(df_text, ["x"])

    phi_text = sections.get("phi", ["auto"])[0]
    phi = build_auto_phi(f, df, a, b) if phi_text.lower() == "auto" else compile_expression(phi_text, ["x"])

    print(f"Точность eps = {eps},  отрезок [{a}, {b}],  x0 = {x0}\n")

    # подсказка для графического определения начального приближения
    print("Отрезки смены знака f(x) на [a, b]:")
    for left, right in find_sign_changes(f, a, b):
        print(f"  [{left:.4f}, {right:.4f}]")
    print()

    # --- условия сходимости простой итерации
    dphi_low, dphi_high = sample_range(lambda x: derivative(phi, x), a, b)
    q = max(abs(dphi_low), abs(dphi_high))
    phi_low, phi_high = sample_range(phi, a, b)
    print("Простая итерация:  q = max|phi'(x)| на [a, b] =", round(q, 6))
    print("  phi отображает [a, b] в себя:", a <= phi_low and phi_high <= b)
    print("  условие q < 1 выполнено:", q < 1, "\n")

    # --- условия сходимости метода Ньютона
    df_low, df_high = sample_range(df, a, b)
    d2f_low, d2f_high = sample_range(lambda x: second_derivative(f, x), a, b)
    print("Метод Ньютона:")
    print("  f'(x) не меняет знак на [a, b]:", df_low * df_high > 0)
    print("  f''(x) не меняет знак на [a, b]:", d2f_low * d2f_high > 0)
    print("  f(x0)*f''(x0) > 0:", f(x0) * second_derivative(f, x0) > 0, "\n")

    # --- решение
    x_ref = reference_root(f, df, x0)
    x_si, k_si, hist_si = simple_iteration(phi, x0, eps, q)
    x_nw, k_nw, hist_nw = newton(f, df, x0, eps)

    print(f"Простая итерация: x = {x_si:.10f},  итераций: {k_si},  |x - x*| = {abs(x_si - x_ref):.3e}")
    print(f"Метод Ньютона:    x = {x_nw:.10f},  итераций: {k_nw},  |x - x*| = {abs(x_nw - x_ref):.3e}")
    print(f"Эталон (Ньютон до 1e-15): x* = {x_ref:.12f},  f(x*) = {f(x_ref):.3e}")
    print("Корень положительный:", x_nw > 0, "\n")

    # --- анализ зависимости погрешности от числа итераций (до предела точности)
    _, _, full_si = simple_iteration(phi, x0, 1e-14, q)
    _, _, full_nw = newton(f, df, x0, 1e-14)
    print_table("Простая итерация: погрешность по итерациям", full_si, x_ref)
    print_table("Метод Ньютона: погрешность по итерациям", full_nw, x_ref)

    save_graph(f, a, b, x0, x_ref, "lab2_1_graph.png")
    save_error_plot(full_si, full_nw, x_ref, "lab2_1_error.png")


if __name__ == "__main__":
    main()
