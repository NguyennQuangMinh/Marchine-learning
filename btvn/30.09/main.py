from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

x_values = np.array([[1.251 ,1.442 ,-2.723 ,-0.161 ,-2.876 ,-2.410 ,-2.364 ,-2.197 ,1.484 ,1.863 ,1.852 ,-0.223 ,2.304 ,-0.601 ,0.592 ,1.777 ,2.071 ,1.149 ,0.222 ,0.914]])
x = x_values.T
y_values = np.array([[5.724 ,6.312 ,-6.124 ,1.504 ,-6.584 ,-5.199 ,-5.107 ,-4.562 ,6.510 ,7.621 ,7.593 ,1.374 ,8.894 ,0.239 ,3.755 ,7.304 ,8.226 ,5.428 ,2.676 ,4.778]])
y = y_values.T
print (x.shape[0])
one =np.ones((x.shape[0],1))
Xbar=np.concatenate((one,x),axis=1)
A = np.dot(Xbar.T,Xbar)
b = np.dot(Xbar.T,y)
w_lr = np.dot(np.linalg.pinv(A),b)
print('Solution found by formula : w =',w_lr.T)
w=w_lr
w_0=w[0][0]
w_1=w[1][0]
x0 = np.linspace(-2,2,2,endpoint=True)
y0=w_0 + w_1*x0
plt.plot(x.T,y.T,'b.')
plt.plot(x0,y0 ,'r',linewidth = 2)
plt.show()