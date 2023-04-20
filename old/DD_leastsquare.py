import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.metrics import mean_squared_error
import cvxpy as cp
# Generate a toy dataset
n_samples = 125
dim = 200
def data(nsamples,ndim):
    X, Y = datasets.make_regression(n_samples=nsamples, n_features=ndim,
                                    n_informative = 0,noise = 0,random_state=None)
    Y = Y * 2 - 1
    #X_Train = X
    #Y_train = np.reshape(Y, (len(Y), 1))
    X_train, X_test, Y_train , Y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=42)
    # r = np.ones(X.shape[0])
    # X = np.insert(X, 0, r, axis=1)
    # X_train = X_train - np.reshape(np.mean(X_train,1), (len(np.mean(X_train,1)), 1))
    X_train = X_train/np.linalg.norm(X_train)
    X_test = X_test / np.linalg.norm(X_test)
    return X_train,X_test, Y_train , Y_test


n_features_list = np.arange(0, dim+1, 1)
costs =[]
testerrsa = np.zeros(n_features_list.shape[0])
trainerrsa = np.zeros(n_features_list.shape[0])
def leastsquare(x_train, y_train):
    # LS
    w_l = np.linalg.pinv(x_train).dot(y_train).ravel()
    return w_l.ravel()
for i in range(0,40,1):
    testerrs = []
    trainerrs =[]
    X_train, X_test, Y_train, Y_test = data(n_samples, dim)
    for feature in n_features_list:
        x=X_train[:,:feature]
        xt=X_test[:,:feature]

        #print(x.shape)
        w_l = leastsquare(x, Y_train)

        testerr = mean_squared_error(xt@w_l , Y_test)
        testerrs.append(testerr)
        trainerr = mean_squared_error(x@w_l , Y_train)
        trainerrs.append(trainerr)
    testerrsa = testerrsa+np.array(testerrs)
    trainerrsa = trainerrsa + np.array(trainerrs)

plt.plot(n_features_list / 100, np.log(testerrsa / 40))
# plt.plot(n_features_list / 100, testerrsa / 40)
plt.plot(n_features_list / 100, trainerrsa / 40*testerrsa / 40)
plt.title("Test Risk vs Dimensions for Well-Specified Model (200 Samples)")
plt.xlabel("# Dimensions/Samples")
# plt.ylim(0,13)
plt.ylabel("Test MSE")
plt.show()