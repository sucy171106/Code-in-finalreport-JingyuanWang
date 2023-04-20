import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
import cvxpy as cp
from matplotlib.colors import ListedColormap
from sklearn import datasets

# Generate a toy dataset
dim = 5000
n_samples = 20
rng = np.random.RandomState(1)
observed = rng.uniform(low=-1, high=1, size=(n_samples, dim))
# print(observed)
labels = np.repeat([-1, 1], int((n_samples + 1) / 2))[:n_samples, None]  # drop last one if necessary
inputs = observed * labels
# print(labels)
costs = []
grads = []


def LogisticLoss(w, X, y, lam):
    # Computes the cost function for all the training samples
    m = X.shape[0]
    Xw = np.matmul(X, w)
    yT = y.reshape(-1, 1)
    yXw = np.multiply(yT, Xw)

    f = np.mean(np.logaddexp(0, -yXw))
    gMul = np.exp(-yXw) / (1 + np.exp(-yXw))
    ymul = -1 * yT * gMul
    g = np.matmul(ymul.reshape(1, -1), X)
    g = g.reshape(-1, 1)
    return [f, g]


def model_logloss(x_train, y_train, num_iterations=20000, learning_rate=0.5):
    w_log = np.zeros((x_train.shape[1], 1))  # GRADED FUNCTION: initialize_with_zeros

    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate

        [cost, dw_log] = LogisticLoss(w_log, x_train, y_train, 1)
        # print(cost)
        # print(dw_log)
        w_log = w_log - learning_rate * dw_log
        # cost = np.linalg.norm(x_train@w_sl - y_train)

        if i % 100 == 0:
            costs.append(cost)
            grads.append((dw_log))
            # Print the cost every 1000 training examples
        if 1 and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return w_log.ravel(), costs, grads


costs = []
grads =[]
w,costvalue,grads = model_logloss(inputs,labels,num_iterations=4000,learning_rate=0.00001)


plt.plot(range(0,4000,100),costvalue)
plt.scatter(range(0,4000,100),costvalue)
plt.rcParams.update({'font.size': 12})
plt.grid()
plt.title("Gradient descent Demo",fontsize=14)
plt.ylabel("Logistic loss",fontsize=12)
plt.xlabel("# num_iterations",fontsize=12)
# plt.axvline(thre,ls='-.', label='Interpolation Threshold',c='#FA7F6F')
plt.savefig('GradientdescentDemo.jpg', dpi=300)