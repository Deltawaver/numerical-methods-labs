from pathlib import Path
import sys
import numpy as np
from lab4_1_cauchy import grid


def shooting(x, left, right):
    from lab4_1_cauchy import rk4_step

    def trajectory(eta):
        a = x[0]
        z = left - 2*eta
        w = (4*eta - 4*a*z)/(2*a+1)
        state = np.array([eta,z,w])
        values = [eta]
        for i in range(len(x)-1):
            state = rk4_step(lambda t,s: np.array([s[1],s[2],-2*s[2]]),
                             x[i],state,x[i+1]-x[i])
            values.append(state[0])
        return np.array(values), state[1]-right

    y0, r0 = trajectory(0.)
    y1, r1 = trajectory(1.)
    if abs(r1-r0) < 1e-12:
        raise ValueError("Невозможно определить параметр стрельбы")
    eta = -r0/(r1-r0)
    y, residual = trajectory(eta)
    return y, eta, residual


def finite_difference(x, left, right):
    h = x[1]-x[0]
    n = len(x)
    matrix = np.zeros((n,n))
    rhs = np.zeros(n)
    matrix[0,:3] = [2-3/(2*h), 2/h, -1/(2*h)]
    rhs[0] = left
    matrix[-1,-3:] = [1/(2*h), -2/h, 3/(2*h)]
    rhs[-1] = right
    for i,t in enumerate(x[1:-1],1):
        matrix[i,i-1:i+2] = [
            (2*t+1)/h**2-2*t/h,
            -2*(2*t+1)/h**2-4,
            (2*t+1)/h**2+2*t/h]
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from lab1.LU import gauss_lu, solve_lu
    p,l,u,_ = gauss_lu(matrix,n)
    return solve_lu(p,l,u,rhs,n)


def main():
    filename = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name("data4.2.txt")
    a,b,h,left,right = np.loadtxt(filename)
    if len(sys.argv)>2:
        h=float(sys.argv[2])
    if not (a == -2 and b == 0):
        raise ValueError("Вариант 8 задан на отрезке [-2, 0]")
    x, fine_x = grid(a,b,h), grid(a,b,h/2)
    exact=3*x+np.exp(-2*x)
    print("4.2. Вариант 8. Ищем гладкое решение через x=-0.5")
    for name,order,solver in [
        ("Стрельба",4,lambda t:shooting(t,left,right)[0]),
        ("Конечные разности",2,lambda t:finite_difference(t,left,right))]:
        y=solver(x)
        fine=solver(fine_x)[::2]
        rr=abs(fine-y)/(2**order-1)
        print(f"\n{name}: x, y, точное, ошибка, РР мелкой сетки")
        for row in zip(x,y,exact,abs(y-exact),rr):
            print(" ".join(f"{v: .10f}" for v in row))
        print(f"Максимальная ошибка: {max(abs(y-exact)):.6g}; РР: {max(rr):.6g}")
    _,eta,residual=shooting(x,left,right)
    print(f"Параметр стрельбы y(-2)={eta:.10f}, невязка y'(0)-1={residual:.3g}")


if __name__ == "__main__":
    main()
