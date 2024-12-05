from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt

def my_cost_function(x):
    return x**2 + 3*x*np.cos(x) - 2*x

def my_gradient(x):
    return 2*x + 3*np.cos(x) - 3*x*np.sin(x) - 2

def my_gradient_descent(eta, initial_x):
    x_values = [initial_x]
    for iteration in range(100):
        x_new = x_values[-1] - eta * my_gradient(x_values[-1])
        if abs(my_gradient(x_new)) < 1e-3:
            break
        x_values.append(x_new)
    return (x_values, iteration)

# Vẽ đồ thị
x_range = np.linspace(-4, 4, 1000)
plt.plot(x_range, my_cost_function(x_range), label='Cost Function')
(x1_values, it1) = my_gradient_descent(0.1, -2)
(x2_values, it2) = my_gradient_descent(0.1, 4)
plt.scatter(x1_values, [my_cost_function(x) for x in x1_values], color='black', label='Initial Point: x = -2', marker='o', s=50)
plt.scatter(x2_values, [my_cost_function(x) for x in x2_values], color='blue', label='Initial Point: x = 4', marker='o', s=50)
plt.legend()
plt.xlabel('x')
plt.ylabel('Cost')
plt.title('Gradient Descent with Cost Function')
plt.grid(True)
plt.show()

print('Solution x1 = %f, cost = %f, obtained after %d iterations' % (x1_values[-1], my_cost_function(x1_values[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations' % (x2_values[-1], my_cost_function(x2_values[-1]), it2))