import numpy as np
import matplotlib.pyplot as plt


def lin(x, k, b):
    return k * x + b


def lin0(x, k):
    return k * x


def lin1(x, k, x0):
    return k * (x0 - x)


def gauss(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / 2 / c ** 2)


def parabola_centered(x, x_c, y_c, a):
    return y_c + a * (x - x_c) ** 2

def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c



def plot_func(func, _x, args, n=100, lw=1, ls='-', color='black', label=None, ax=plt):
    x = np.linspace(min(_x), max(_x), n)
    ax.plot(x, func(x, *args), color=color, lw=lw, ls=ls, label=label)


def lsm(X, Y, plot=True):
    n = len(X)
    mx = my = mxy = mx2 = my2 = 0
    for i in range(n):
        mx += X[i]
        my += Y[i]
        mxy += X[i] * Y[i]
        mx2 += X[i] ** 2
        my2 += Y[i] ** 2
    mx /= n
    my /= n
    mxy /= n
    mx2 /= n
    my2 /= n

    k = (mxy - mx * my) / (mx2 - mx ** 2)
    b = my - k * mx
    sk = np.sqrt(1 / (n - 2)) * np.sqrt((my2 - my ** 2) / (mx2 - mx ** 2) - k ** 2)
    sb = sk * np.sqrt(mx2)

    if plot:
        plot_func(lin, X, (k, b))
        plt.scatter(X, Y)

    return [[k, b], [sk, sb]]


def par(x, a):
    return a * x ** 2


def plot_par(_X, a):
    x1 = min(_X)
    x2 = max(_X)
    X = np.linspace(x1, x2, 3)
    Y = a * X ** 2
    plt.plot(X, Y, color='black')


def lsm2(X, Y, plot=True):
    n = len(X)
    mx2y = mx4 = 0
    for i in range(n):
        mx2y += X[i] ** 2 * Y[i]
        mx4 += X[i] ** 4
    mx2y /= n
    mx4 /= n

    a = mx2y / mx4

    if plot:
        plot_par(X, a)
        plt.scatter(X, Y)

    return a
