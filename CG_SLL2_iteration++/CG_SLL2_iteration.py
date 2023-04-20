import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import cvxpy as cp

# define a toy dataset  or "regression"
set = "classification"

if set == "classification":
    n_samples = 16
    dim = 25
else:
    n_samples = 16
    dim = 25

n_features_list = np.arange(3, dim + 1, 10)
costs =[]
distances2 = []

# plot section
# def pltb(set):
#     if set == "classification":
#         # plot the training point
#         idx0 = np.where(Y_train == -1)
#         idx1 = np.where(Y_train == 1)
#         plt.plot(X_Train[idx0[0], 0], X_Train[idx0[0], 1], 'rx')
#         plt.plot(X_Train[idx1[0], 0], X_Train[idx1[0], 1], 'bo')
#     else:
#         plt.plot(X_Train[ 0], X_Train[ 1], 'rx')
#     xp = np.linspace(np.min(X_Train[:, 0]), np.max(X_Train[:, 0]), 100)
#
#     # plot the decision boundary
#     yp = - (w_s[1] * xp+ w_s[0]) / w_s[2]
#     plt.plot(xp, yp, '--b', label='L2')
#
#     # plot the decision boundary
#     yp1 = - (w_sl[1] * xp+ w_sl[0]) / w_sl[2]
#     plt.plot(xp, yp1, '-r', label='sl')
#     plt.legend()
#     plt.show()


# generate the data classification or regression
def data(set):
    if set == "classification":
        X, Y = datasets.make_blobs(n_samples=n_samples, n_features=dim, centers=2,
                                   cluster_std=[1.0, 1.0], random_state=42)
        Y = Y * 2 - 1
        X_Train = X
        Y_train = np.reshape(Y, (len(Y), 1))
    else:
        X, Y = datasets.make_regression(n_samples=n_samples, n_features=dim, random_state=42)
        X_Train = X
        Y_train = np.reshape(Y, (len(Y), 1))
    return X_Train,Y_train

# least square model
def leastsquare(x_train, y_train):
    # LS
    w_l = np.linalg.pinv(x_train).dot(y_train).ravel()
    return w_l.ravel()

# minimum L2 norm model
def L2(x_train, y_train):

    n = x_train.shape[1]
    s_s = cp.Variable((n, 1))

    objective = cp.Minimize(cp.norm(s_s))
    constraints = [y_train == x_train @ s_s]
    prob = cp.Problem(objective, constraints)

    prob.solve()
    s_s_value = s_s.value

    return s_s_value.ravel()

# GRADED FUNCTION: model
def model_squreloss(x_train, y_train, num_iterations=2000, learning_rate=0.5):
    w_sl = np.zeros((x_train.shape[1], 1)) # GRADED FUNCTION: initialize_with_zeros
    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate
        dw_sl = -(x_train.T@ (y_train - x_train@w_sl ))/n_samples
        # update rule
        w_sl = w_sl - learning_rate * dw_sl
        cost = (np.linalg.norm(y_train-x_train@w_sl )**2)/n_samples

        # Print the cost every 1000 training examples
        if i % 10 == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))
        if 1 and i % 10== 0:
            distance2 = np.linalg.norm(w_sl.ravel() - w_s)
            distances2.append(distance2)
    return w_sl.ravel(),distances2
k=10
# generate the data
X_Train,Y_train =data(set)

# train with L2 model
w_s = L2(X_Train, Y_train)

# train with SL model
w_sl,distances2 = model_squreloss(X_Train, Y_train, num_iterations = 100000, learning_rate = 0.0004)


# plot section
# pltb(set)
distance2 = np.linalg.norm(w_sl - w_s)
print(distance2)
plt.plot(np.arange(0,100000,k),distances2,c='#2878B5')
plt.xlabel("# Iterations")
plt.ylabel("Distances")
plt.title("Distances between SL and L2 vs Iterations")

# plt.legend()
plt.grid()
# plt.xscale('log')
plt.savefig('SL_L2_Iterations.jpg', dpi=300)
plt.show()




