import numpy as np
import matplotlib.pyplot as plt

# Khai báo data dưới dạng mảng NumPy
data = np.array([[x1, y1], [x2, y2], ...])

# Các phần còn lại của mã không cần thay đổi


# xây dựng hàm loss
def loss_function(data,theta):
    #get m and b
    m = theta[0]
    b = theta[1]
    loss = 0
    #on each data point
    for i in range(0, len(data)):
        #get x and y
        x = data[i, 0]
        y = data[i, 1]
        #predict the value of y
        y_hat = (m*x + b)
        #compute loss as given in quation (2)
        loss = loss + ((y - (y_hat)) ** 2)
    #mean sqaured loss
    mean_squared_loss = loss / float(len(data))
    return mean_squared_loss

# xây dựng hàm tính đạo hàm
def compute_gradients(data, theta):
    gradients = np.zeros(2)
    #total number of data points
    N = float(len(data))
    m = theta[0]
    b = theta[1]
    #for each data point
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        #gradient of loss function with respect to m as given in (3)
        gradients[0] += - (2 / N) * x * (y - (( m* x) + b))
        #gradient of loss funcction with respect to b as given in (4)
        gradients[1] += - (2 / N) * (y - ((theta[0] * x) + b))
    #add epsilon to avoid division by zero error
    epsilon = 1e-6
    gradients = np.divide(gradients, N + epsilon)
    return gradients




# Các phần còn lại của mã không cần thay đổi

# Thực hiện thuật toán
theta = np.zeros(2)
gr_loss = []
num_iterations = 1000  # Số vòng lặp giảm xuống để kiểm tra hội tụ nhanh hơn
learning_rate = 1e-2

for t in range(num_iterations):
    # compute gradients
    gradients = compute_gradients(data, theta)
    # update parameter
    theta = theta - (learning_rate * gradients)
    # store the loss
    gr_loss.append(loss_function(data, theta))

plt.plot(gr_loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

print('Final theta:', theta)
