import numpy as np
import sympy as syp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import regressions as reg

angle = [1892, 1898, 1938, 2196, 2212,
         2246, 2258, 2288, 2306, 2318,
         2336, 2346, 2370, 2390, 2404,
         2420, 2438, 2444, 2482, 2492,
         2514, 2540, 2552, 2616, 2646]

lam = [5331, 5341, 5401, 5852, 5882,
       5945, 5976, 6030, 6074, 6096,
       6143, 6164, 6217, 6267, 6305,
       6334, 6383, 6402, 6507, 6533,
       6599, 6678, 6717, 6929, 7032]

x, lam0, C, d0 = syp.symbols('x lam0 C d0')

func = lam0 + C / (x - d0)

cs = reg.plot_func(func, lam, angle, y_err=2)
y, sy = reg.eval_func(func, cs, 6000)
print(y, sy)

plt.show()


