import matplotlib.pyplot as plt
import numpy as np
import math
def grad(x):
    return 3*x*x + 6*x + 3*np.sin(x) + 8*np.cos(x)
def cost(x):
    return x*x*x + 3*x*x - 3*np.cos(x)+ 8*np.sin(x)
def myGD1(eta, x0):
    x = [x0]
    for i in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x,i)

from matplotlib.backends.backend_pdf import PdfPages
def draw_grid(x1, ids, filename, nrows = 2, ncols = 3, start= -5):
    x0 = np.linspace(start,1,500)
    y0 = cost(x0)
    width = 4 * ncols
    height = 4 * nrows
    plt.close('all')
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, height))
    with PdfPages(filename) as pdf:
        for i, k in enumerate(ids):
            r = i // ncols
            c = i % ncols
            x = x1[ids[i]]
            y = cost(x)
            str0 = 'a {}/{} , b ={:.3f}'.format(ids[i], len(x1)-1, grad(x))
            axs[r][c].plot(x0, y0, 'r')
            axs[r][c].set_xlabel(str0, fontsize=10)
            axs[r][c].plot(x, y, 'yo', markersize=10, markerfacecolor='b')
            axs[r][c].plot()

            axs[r][c].tick_params(axis='both', which='major')
        pdf.savefig(bbox_inches='tight')
        plt.show()
#good learning rate
filename = 'Thai.pdf'
(x1,i1) = myGD1(.3, -2)
(x2,i2) = myGD1(.3, 1)
print('x1 = %f , cost = %f, obtain after %d iterations' %(x1[-1], cost(x1[-1]), i1))
print('x2 = %f , cost = %f, obtain after %d iterations' %(x2[-1], cost(x2[-1]), i2))
ids = [0, 5, 10, 15, 20, 25]
draw_grid(x1, ids, filename)