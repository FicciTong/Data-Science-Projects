# Import libraries
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

# Non-regularized Multiclass Logistic Regression
# Loading the data and defining the training set.
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
print(X.shape)
print(y.shape)

# Plotting the training set
pos = y == 1
neg = y == 0
fig, ax = plt.subplots()
ax.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')
ax.scatter(X[neg, 0], X[neg, 1], marker='o', c='y')
ax.set_xlabel('Exam 1 score')
ax.set_ylabel('Exam 2 score')
ax.legend(('Admitted', 'Not admitted'))
plt.show()

# Compute cost and gradient
m, n = X.shape
X = np.concatenate((np.ones((m, 1)), X), axis=1)
initial_theta = np.zeros((n+1, 1))

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define cost function that returns a dictionary conataining cost and gradient
def costFunction(theta, X, y):
    m = y.shape[0]
    J = 0
    grad = np.zeros(theta.shape)
    J = (1./ m) * (np.dot((-y.T), np.log(sigmoid(np.dot(X, theta)))) - np.dot((1 - y.T), np.log(1 - sigmoid(np.dot(X, theta)))))
    grad = (1 / m) * np.dot(X.T, (sigmoid(np.dot(X, theta)).reshape(100,) - y).T)
    output_dict = {
        'J': J,
        'grad': grad
    }
    return output_dict

result1 = costFunction(initial_theta, X, y);

print('Cost at initial theta (zeros): \n', result1['J']);
print('Expected cost (approx): 0.693\n');
print('Gradient at initial theta (zeros): \n');
print(' \n', result1['grad']);
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2]);
result2 = costFunction(test_theta, X, y);

print('\nCost at test theta:\n', result2['J']);
print('Expected cost (approx): 0.218\n');
print('Gradient at test theta: \n');
print('\n', result2['grad']);
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');
