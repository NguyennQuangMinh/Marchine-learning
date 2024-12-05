from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt

def my_gradient(x):
    return 2*x + 3*np.cos(x) - 2

def my_cost_function(x):
    return x**2 + 3*x*np.cos(x) - 2*x

def my_gradient_descent_momentum_nag(learning_rate, momentum_factor, x0):
    x_values = [x0]
    momentums = [0]  # Tốc độ của momentum
    for iteration in range(100):
        x_pred = x_values[-1] - learning_rate * momentum_factor * momentums[-1]
        momentum_new = momentum_factor * momentums[-1] + learning_rate * my_gradient(x_pred)
        x_new = x_values[-1] - momentum_new
        if abs(my_gradient(x_new)) < 1e-3:
            break
        x_values.append(x_new)
        momentums.append(momentum_new)
    return (x_values, iteration)

# Vẽ đồ thị hàm số và quá trình tối ưu hóa với Momentum + NAG
x_range = np.linspace(-4, 8, 1000)
plt.plot(x_range, my_cost_function(x_range), label='Cost Function')
(x1_values_momentum_nag, it1_momentum_nag) = my_gradient_descent_momentum_nag(0.1, 0.9, -2)
(x2_values_momentum_nag, it2_momentum_nag) = my_gradient_descent_momentum_nag(0.1, 0.9, 4)
plt.scatter(x1_values_momentum_nag, [my_cost_function(x) for x in x1_values_momentum_nag], color='black', label='Initial Point: x = -2 (Momentum and NAG)', marker='o', s=50)
plt.scatter(x2_values_momentum_nag, [my_cost_function(x) for x in x2_values_momentum_nag], color='blue', label='Initial Point: x = 4 (Momentum and NAG)', marker='o', s=50)
plt.legend()
plt.xlabel('x')
plt.ylabel('Cost')
plt.title('Gradient Descent with Momentum and NAG')
plt.grid(True)
plt.show()

print('Solution x1 (Momentum + NAG) = %f, cost = %f, obtained after %d iterations' % (x1_values_momentum_nag[-1], my_cost_function(x1_values_momentum_nag[-1]), it1_momentum_nag))
print('Solution x2 (Momentum + NAG) = %f, cost = %f, obtained after %d iterations' % (x2_values_momentum_nag[-1], my_cost_function(x2_values_momentum_nag[-1]), it2_momentum_nag))