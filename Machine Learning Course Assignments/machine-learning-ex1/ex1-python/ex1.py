import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading the data for ex1
f = open('ex1data1.txt', mode='r')
data = f.read()
data = data.split('\n')
data = data[:len(data) - 1]

# Assign X and y values
X = []
y = []
rows = []
for i in data:
    temp = i.split(',')
    temp[0] = float(temp[0])
    temp[1] = float(temp[1])
    X.append(temp[0])
    y.append(temp[1])
    rows.append(temp)

# Plot the training set data
plt.scatter(X, y, marker='x', c='r')
plt.xlim(4, 24)
plt.ylim(-5, 25)
plt.xlabel('Population Size in 10000s')
plt.ylabel('Profit in $10000s')
plt.show()

# Adding the ones column (X0) to X
X = np.matrix(X)
X = np.transpose(X)
ones_col = np.transpose(np.matrix(np.ones(np.shape(X)[0])))
X = np.concatenate((ones_col, X), axis=1)
print(X)

# Initialize theta value
theta = np.matrix(np.zeros((2, 1)))

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# Defining the cost function
def computeCost(X, y, theta):
    m = np.shape(y)[0]
    J = 0
    for i in range(m):
        J += (np.dot(X[i, :], theta) - y[i]) ** 2
    J = J/2/m
    return J

print('\nTesting the cost function ...\n')
# Compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = ', J.item(0, 0));
print('Expected cost value (approx) 32.07\n');

# further testing of the cost function
J = computeCost(X, y, np.matrix('-1; 2'));
print('\nWith theta = [-1 ; 2]\nCost computed = ', J.item(0, 0));
print('Expected cost value (approx) 54.24\n');

# Defining the function to perform gradient descent
def gradientDescent(X, y, theta, alpha, num_iters):
    m = np.shape(y)[0]
    J_history = []
    for iter in range(num_iters):
        delta = np.dot((np.dot(np.transpose(theta), np.transpose(X)) - np.transpose(y)), X)
        delta = np.transpose(delta)
        theta = theta - (alpha / m) * delta
        J_history.append(computeCost(X, y, theta).item(0, 0))
    output_dict = {
        'theta': theta,
        'J_history': J_history
    }
    return output_dict

print('\nRunning Gradient Descent ...\n')
# run gradient descent
output = gradientDescent(X, y, theta, alpha, iterations);
theta = output['theta']

# print theta to screen
print('Theta found by gradient descent:\n', theta.item(0), '\n', theta.item(1));
print('Expected theta values (approx)');
print(' -3.6303\n  1.1664\n\n');
