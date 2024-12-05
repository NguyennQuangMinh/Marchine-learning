from __future__ import division, print_function , unicode_literals
import  numpy as np
import  matplotlib
import matplotlib.pyplot as plt

def cost(x):
    return x**3 + 3*x**2 - 3*np.cos(x) + 8*np.sin(x)
def grad(x):
    return 3*x**2 + 6*x + 3*np.sin(x) + 8*np.cos(x)

def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)
(x1, it1) = myGD1(0.2, .5)
(x2, it2) = myGD1(0.2, .3)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))