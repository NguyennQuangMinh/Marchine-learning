import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

N = 100
d0 = 2
C = 3
X = np.zeros((d0, N * C))
y = np.zeros(N * C, dtype='uint8')

for j in range(C):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
    X[:, ix] = np.c_[r * np.sin(t), r * r * np.cos(2*t)].T
    y[ix] = j

# Hiển thị dữ liệu:
plt.plot(X[0, :N], X[1, :N], 'bs', markersize=7)
plt.plot(X[0, N:2 * N], X[1, N:2 * N], 'ro', markersize=7)
plt.plot(X[0, 2 * N:], X[1, 2 * N:], 'g^', markersize=7)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
plt.savefig('Anh.png', bbox_inches='tight', dpi=600)
plt.show()

def softmax(V):
    e_V = np.exp(V - np.max(V, axis=0, keepdims=True))
    Z = e_V / e_V.sum(axis=0)
    return Z

def convert_labels(y, C=3):
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))),
                          shape=(C, len(y))).toarray()
    return Y

def cost(Y, Yhat):
    return -np.sum(Y * np.log(Yhat)) / Y.shape[1]


d0 = 2
d1 = h = 50
d2 = 50
d3 = C = 3

W1 = 0.01 * np.random.randn(d0, d1)
b1 = np.zeros((d1, 1))
W2 = 0.01 * np.random.randn(d1, d2)
b2 = np.zeros((d2, 1))
W3 =0.01 * np.random.randn(d2, d3)
b3 = np.zeros((d3, 1))

Y = convert_labels(y, C)
N = X.shape[1]
eta = 1

for i in range(10000):
    Z1 = np.dot(W1.T, X) + b1
    A1= np.tanh(Z1)
    Z2 = np.dot(W2.T, A1) + b2
    A2 = np.tanh(Z2)
    Z3 = np.dot(W3.T, A2) + b3
    A3 = np.tanh(Z3)
    Yhat = softmax(Z3)

    if i % 1000 == 0:
        loss = cost(Y, Yhat)
        print("vòng lặp %d, mất mát: %f" % (i, loss))

    E3 = (Yhat - Y) / N
    dW3 = np.dot(A2, E3.T)
    db3 = np.sum(E3, axis=1, keepdims=True)
    E2 = np.dot(W3, E3)
    E2 = E2*(1 - np.power(A2, 2))
    dW2 = np.dot(A1, E2.T)
    db2 = np.sum(E2, axis=1, keepdims=True)

    E1 = np.dot(W2, E2)
    E1 = E1 * (1 - np.power(A1, 2))
    dW1 = np.dot(X, E1.T)
    db1 = np.sum(E1, axis=1, keepdims=True)

    W1 += -eta * dW1
    b1 += -eta * db1
    W2 += -eta * dW2
    b2 += -eta * db2
    W3 += -eta * dW3
    b3 += -eta * db3

Z1 = np.dot(W1.T, X) + b1
A1 = np.tanh(Z1)
Z2 = np.dot(W2.T, A1) + b2
A2 = np.tanh(Z2)
Z3 = np.dot(W3.T, A2) + b3
predicted_class = np.argmax(Z3, axis=0)
print(' accuracy: %.2f %%' % (100 * np.mean(predicted_class == y)))