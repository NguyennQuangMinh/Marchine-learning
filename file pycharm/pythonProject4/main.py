from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def grad(x):
    return 2*x + 10*np.cos(x)
def cost(x):
    return  x**2 + 10*np.sin(x)
def has_converged(x_new,grad):
    return np.linalg.norm(grad(x_new))< 1e-3
def GD_momentum(x_init,grad,eta,gamma):
    x = [x_init]
    v_old = np.zeros_like(x_init)
    for it in range(100):
        v_new = gamma*v_old + eta*grad(x[-1])
        x_new = x[-1]- v_new
        if has_converged(x_new,grad):
            break
        x.append(x_new)
        v_old=v_new
    return  x,it

def myGD1(x0, eta):
  x = [x0]
  for i in range(100):
    x_new = x[-1] - eta*grad(x[-1])
    if abs(grad(x_new))< 1e-3:
       break
    x.append(x_new)
  return(x, i)
(x,i) = myGD1(5, 0.1)
print('Solution GD xmin =  %f, cost =  %f, obtained after %d iterations' %(x[-1], cost(x[-1]), i))
def draw_grid(x, ids, filename, nrows=2, ncols=3, start=-8):
    x0 = np.linspace(-4, 6, 1000)
    y0 = cost(x0)
    width = 4 * ncols
    height = 4 * nrows
    plt.close('all')
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, height))
    with PdfPages(filename) as pdf:
        for i, k in enumerate(ids):
            r = i // ncols
            c = i % ncols
            x_i = x[ids[i]]
            y_i = cost(x_i)
            str0 = 'iter {}/{}, grad {:.3f}'.format(ids[i],len(x) - 1 , grad(x_i))
            axs[r][c].plot(x0, y0, 'b')
            axs[r][c].set_xlabel(str0, fontsize=13)
            axs[r][c].plot(x_i, y_i, 'bo', markersize=10, markerfacecolor='r')
            axs[r][c].plot()
            axs[r][c].tick_params(axis='both', which='major', labelsize=15)
        pdf.savefig(bbox_inches='tight')
        plt.show()
filename= 'momentum.pdf'
x, it= GD_momentum(5,grad,0.1,0.9)
print('Solution GD_momentum xmin = %f, cost = %f, obtained after %d iterations'%(x[-1], cost(x[-1]), it))
ids=[4,99]
draw_grid(x,ids,filename)
