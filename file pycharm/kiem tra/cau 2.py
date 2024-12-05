import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2)
X = np.array([[10, 12, 14, 15, 18, 19, 20, 21, 22, 24, 25, 27, 28, 30, 32, 33, 35, 36, 37, 40]])
y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):
    w = [w_init]
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta * (yi - zi) * xi
            count += 1
        if count % check_w_after == 0:
            if len(w) > check_w_after and np.linalg.norm(w_new - w[-check_w_after]) < tol:
                return w
        w.append(w_new)
    return w

eta = 0.05
d = X.shape[0]
w_init = np.random.randn(d, 1)
w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w[-1])
print(sigmoid(np.dot(w[-1].T, X)))

x0 = X[1, np.where(y == 0)][0]
y0 = y[np.where(y == 0)]
x1 = X[1, np.where(y == 1)][0]
y1 = y[np.where(y == 1)]

plt.plot(x0, y0, 'ro', markersize=8)
plt.plot(x1, y1, 'bs', markersize=8)
xx = np.linspace(0, 6, 1000)
w0 = w[-1][0][0]
w1 = w[-1][1][0]
threshold = -w0 / w1
yy = sigmoid(w0 + w1 * xx)
plt.axis([2, 8, -1, 2])
plt.plot(xx, yy, 'g-', linewidth=2)
plt.plot(threshold, 0.5, 'y', markersize=8)
plt.xlabel('nhiet do')
plt.ylabel('quyet dinh di hoc')
plt.show()