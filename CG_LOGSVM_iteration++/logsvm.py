import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import cvxpy as cp
import scipy as sc

# Generate a toy dataset
n_samples = 10
dim =2000

X_train, Y = datasets.make_blobs(n_samples=n_samples, n_features=dim, centers=2, cluster_std=[2.0, 2.0], random_state=42)
Y = Y * 2 - 1 # change Y to -1,1
Y_train = np.reshape(Y, (len(Y), 1))

r = np.ones(X_train.shape[0])

# Calculate the maximum singular value
ss = sc.sparse.linalg.svds(X_train)

s0 = np.max(ss[0])
s1 = np.max(ss[1])
s2 = np.max(ss[2])

lr = 1/(s1**2)

n_features_list = np.arange(3, dim+1, 10)
costs =[]
distances2 = []

# SVM model
def SVM(x_train, y_train):
    # SVM
    s_s = cp.Variable((x_train.shape[1], 1))

    objective = cp.Minimize(cp.norm(s_s) ** 2)
    constraints = [cp.multiply(y_train, x_train @ s_s) >= 1]
    prob = cp.Problem(objective, constraints)

    prob.solve()
    s_s_value = s_s.value
    return s_s_value


# log loss function
def LogisticLoss(w, X, y, lam):
    # Computes the cost function for all the training samples
    m = X.shape[0]
    Xw = np.matmul(X,w)
    yT = y.reshape(-1,1)
    yXw = np.multiply(yT,Xw)

    f = np.sum(np.logaddexp(0, -yXw))
    gMul = np.exp(-yXw)/(1 + np.exp(-yXw))
    ymul = -1*yT*gMul

    g = np.matmul(ymul.reshape(1, -1), X)

    g = g.reshape(-1, 1)
    return [f, g]


# log model
def model_logloss(x_train, y_train, num_iterations=20000, learning_rate=0.5):
    w_log = np.zeros((x_train.shape[1], 1))  # GRADED FUNCTION: initialize_with_zeros
    cost1 = 1
    # Gradient descent
    # GRADED FUNCTION: optimize
    i = 0

    while(1):
        # GRADED FUNCTION: propagate
        i= i+1

        [cost, dw_log] = LogisticLoss(w_log, x_train, y_train, 1)
        w_log = w_log - learning_rate * dw_log

        if i % 100000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            if (cost<5e-1):
                 break
    return w_log

# train function
def train(lr):
    # train with log model
    w_log = model_logloss(X_train, Y_train, num_iterations=500000, learning_rate=lr*0.001)

    # train with SVM model
    w_s = SVM(X_train, Y_train)

    # normalize the weights
    w_log_norm = w_log/np.linalg.norm(w_log)
    w_s_norm = w_s/np.linalg.norm(w_s)

    distance2 = np.linalg.norm(w_log_norm-w_s_norm)
    print(distance2)

train(lr)


