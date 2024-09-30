import numpy as np
import sympy as syp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


x, a, b, c = syp.symbols('x a b c')
linear = a * x + b
linear0 = a * x
parabolic = a * x ** 2 + b * x + c


def plot_func(func, x_points, y_points, x_err=None, y_err=None,
              scatter_params=('black', None, 10), line_params=('black', None, 1, '-'), ax=plt, n=100):

    # Приводим функцию к нормальному виду
    p_used = list(func.free_symbols)
    func_np = syp.lambdify(p_used, func, 'numpy')

    # Производим фит
    cs = curve_fit(func_np, x_points, y_points, sigma=y_err)

    # Плоттим графики
    x_range = np.linspace(min(x_points), max(x_points), n)
    ax.plot(x_range, func_np(x_range, *cs[0]))
    ax.errorbar(x_points, y_points, xerr=x_err, yerr=y_err, fmt='o')

    return cs


def eval_func(func, p_vals, x_val):
    # Распаковка результата curve_fit()
    p_opt, p_cov = p_vals[0], p_vals[1]
    p_used = list(func.free_symbols)

    # Словарь подстановок
    subs = dict(zip(p_used[1:], p_opt))
    subs[p_used[0]] = x_val

    # Значение с оптимальными параметрами
    y = float(func.evalf(subs=subs))

    # Частные производные в x_val
    derivatives = np.array([])
    for p in p_used[1:]:
        der = func.diff(p)
        der_eval = float(der.evalf(subs=subs))
        derivatives = np.append(derivatives, der_eval)

    # Трюки с матрицей ковариаций чтобы посчитать вот это:
    # sx^2 = <(df/dx1 * sx1 + ... + df/dxn * sxn)^2>
    outer = np.outer(derivatives, derivatives)
    sy2 = np.sum(outer * p_cov)

    # Значение и погрешность
    return y, np.sqrt(sy2)
