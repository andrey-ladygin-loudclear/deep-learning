from scipy import optimize

def f(x):
    return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2

print f([1,1])

result = optimize.brute(f, ((-5, 5), (-5, 5)))
print result

print optimize.differential_evolution(f, ((-5, 5), (-5, 5)))

import numpy as np
def g(x):
    return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))

print optimize.check_grad(f, g, [2, 2])
print optimize.fmin_bfgs(f, [2, 2], fprime=g)
print optimize.minimize(f, [2, 2])
print optimize.minimize(f, [2, 2], method='BFGS', jac=g)
print optimize.minimize(f, [2, 2], method='Nelder-Mead')
