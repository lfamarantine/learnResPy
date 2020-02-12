# OPTIMIZATION PROBLEMS ----------------------------------------------------------------------

# linear optimisation system example: 75x + 50y + 35z
# quadratic optimisation system example: 2x^2 + 3y^2 + 4z -> markowitz, where quadratic objective is portfolio variance

# covnex vs non-convex: easy to solve or not (non-convex)

# linear optimisation ------------------------
from PIL import Image
# goal is to optimise the following equation system..
Image.open('img/optim_problem.png').show()

from scipy.optimize import linprog
c = [-1, 4]
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0_bounds = (None, None)
x1_bounds = (-3, None)
res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
# using simplex algo..
res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='revised simplex')
print(res)

# what optimisation methods are there?
help(linprog)
# default: interior-point, simplex, revised simplex

import scipy as sp
help(sp.optimize)


















