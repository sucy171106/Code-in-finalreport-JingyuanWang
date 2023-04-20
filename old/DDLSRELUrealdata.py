from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 数据获取
def get_data():
    iris = load_iris()
    data = iris.data
    result = iris.target
    return data, result


# 将数据处理为训练集和测试集
def data_deal(data, result):
    data_list = []
    for i in data:
        tem_list = [i[0], i[1]]
        data_list.append(tem_list)
    res_list = []
    for j in result:
        res_list.append(j)
    train_list = data_list[0: 10] + data_list[20: 80] + data_list[90: 100]
    train_result = res_list[0: 10] + res_list[20: 80] + res_list[90: 100]
    test_list = data_list[0: 40] + data_list[60: 100]
    test_result = res_list[0: 40] + res_list[60: 100]

    return data_list, train_list, test_list, train_result, test_result


X, Y = get_data()
# r = np.ones(X.shape[0])
# X = np.insert(X, 0, r, axis=1)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.4, random_state=52)
#
# print(X.shape)
# print(Y.shape)
# print(trainX.shape)
# print(testX.shape)
# print(trainY.shape)
# print(testY.shape)

v = []
N_relu = 140


# print(v.shape)
def leastsquare(x_train, y_train):
    # LS
    w_l = np.linalg.pinv(x_train).dot(y_train).ravel()
    return w_l.ravel()


def tran(n_features, X_train, X_test):
    vn = v[:n_features, :]
    Z = np.maximum(vn @ X_train.T, 0)
    Zt = np.maximum(vn @ X_test.T, 0)

    # npZ_1 = (2 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))) - 1
    # npZt_1 = (2 * (Zt - np.min(Zt)) / (np.max(Zt) - np.min(Zt))) - 1
    # return npZ_1.T, npZt_1.T


    return Z.T,Zt.T
k = np.arange(2, N_relu, 1)

testerrsa = np.zeros(N_relu - 2)
trainerrsa = np.zeros(N_relu - 2)
wl = []
print(trainX[0:3, 0:3])

for i in range(100):
    v = []
    testerrs = []
    trainerrs = []
    for n_features in np.arange(0, N_relu, 1):
        # vi = np.random.normal(0, 0.05, size=dim)
        vi = np.random.uniform(-1, 1, size=X.shape[1])
        v.append(vi)
    v = np.array(v)
    for n_features in k:
        trainX1, testX1 = tran(n_features, trainX, testX)
        #         print(trainX1.shape)
        #         print(testX1.shape)
        wls = leastsquare(trainX1, trainY)
        trainpredict = trainX1 @ wls
        trainerr = np.mean((trainpredict - trainY) ** 2)
        trainerrs.append(trainerr)

        testpridict = testX1 @ wls
        testerr = np.mean((testpridict - testY) ** 2)
        testerrs.append(testerr)
    testerrsa = testerrsa + np.array(testerrs)
    trainerrsa = trainerrsa + np.array(trainerrs)

plt.plot(k, trainerrs)
plt.plot(k, testerrs)
plt.show()


