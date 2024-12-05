from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt

N = 100 # mau
d0 = 2 # chieu
C = 3 #  lop
X = np.zeros((d0, N*C)) # Một ma trận gồm d0 hàng và N * C cột, được sử dụng để lưu trữ dữ liệu.
y = np.zeros(N*C, dtype='uint8') # Một mảng chứa nhãn lớp của các điểm dữ liệu.

for j in range(C):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # một dãy giá trị góc theta, được tạo bằng cách tạo một dãy tuyến tính từ j * 4 đến (j + 1) * 4 và thêm nhiễu ngẫu nhiên.
  X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T # được cập nhật với các điểm dữ liệu dựa trên r và t, sau đó, y được cập nhật với nhãn lớp.
  y[ix] = j

#ve do thi
plt.plot(X[0,:N], X[1,:N], 'bs', markersize = 7);
plt.plot(X[0,N:2*N], X[1,N:2*N], 'ro', markersize = 7);
plt.plot(X[0,2*N:], X[1,2*N:], 'g^', markersize = 7);
# chỉnh đồ thị
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

#tạo vùng màu
plt.savefig('plong.jpg', bbox_inches='tight', dpi = 800)
plt.show()

def softmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

#one-hot coding
from scipy import sparse
def convert_labels(y, C = 3):
    Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

# cost or loss function  hàm chi phí , hàm tổn hao
def cost(Y, Yhat):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]
d0 = 2
d1 = h = 100
d2 = C = 3
 #Tạo ma trận trọng số ngẫu nhiên
W1 = 0.01*np.random.randn(d0, d1) #d0 : d1 (số hàng : số cột)
b1 = np.zeros((d1, 1))
W2 = 0.01*np.random.randn(d1, d2) #d0 : d2 (số hàng : số cột)
b2 = np.zeros((d2, 1))
#
Y = convert_labels(y, C)
N = X.shape[1]# số lượng mẫu = cột của ma trận X
eta = 1 # learning rate
for i in range(10000):

    Z1 = np.dot(W1.T, X) + b1
    A1 = np.maximum(Z1, 0)  #đầu vào z1 , đầu ra A1
    Z2 = np.dot(W2.T, A1) + b2

    Yhat = softmax(Z2)
    # compute the loss: average cross-entropy loss
    loss = cost(Y, Yhat)
    # print loss after each 1000 iterations
    if i %1000 == 0:
        print("iter %d , loss: %f" %(i, loss))
    # backpropagation
    E2 = (Yhat - Y )/N
    dW2 = np.dot(A1, E2.T)
    db2 = np.sum(E2, axis = 1, keepdims = True)
    E1 = np.dot(W2, E2)
    E1[Z1 <= 0] = 0 # gradient of ReLU
    dW1 = np.dot(X, E1.T)
    db1 = np.sum(E1, axis = 1, keepdims = True)
    #
    W1 += -eta*dW1
    b1 += -eta*db1
    W2 += -eta*dW2
    b2 += -eta*db2

Z1 = np.dot(W1.T, X) + b1
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
predicted_class = np.argmax(Z2, axis=0)
print('training accuracy: %.2f %%' % (100*np.mean(predicted_class == y)))
