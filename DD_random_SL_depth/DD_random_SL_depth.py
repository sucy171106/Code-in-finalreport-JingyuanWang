import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Generate a toy dataset
def data(nsamples, ndim):
    X, Y = datasets.make_regression(n_samples=nsamples, n_features=ndim,
                                    n_informative=ndim, noise=0, random_state=None)

    X = X / np.std(X)    # rescale, make its standard deviation to zero
    Y = Y - np.mean(Y)   # make its mean to zero
    Y = Y / np.std(Y)    # rescale, make its standard deviation to zero
    Y = Y + np.random.normal(0, 0.1, Y.shape)   # add noise to the label

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # split the data
    return X_train, X_test, Y_train, Y_test

# least squares model function
def leastsquare(x_train, y_train):
    # LS
    w_l = np.linalg.pinv(x_train).dot(y_train).ravel()
    return w_l.ravel()


# rescale function for the depth
def sig(x):
    x_ = x
    idx = x_<0.3
    x_[idx]=0.3
    return 2/(1+np.exp(-x_))-1

# the number of the samples is 400, the dimension size is 200
n_samples = 400
dim = 200

# feature size changes from 10 to 200, sample size changes from 10 to 200
n_features_list = np.arange(10, dim + 1, 1)
n_sampels_list = np.arange(10, 200 + 1, 1)

# restore the results from different times
testerrsa_ = np.zeros((n_sampels_list.shape[0], n_features_list.shape[0]))

for i in range(10):
    X_train, X_test, Y_train, Y_test = data(n_samples, dim)
    testerrsa = []

    # for loop changing sample size
    for n_sample in n_sampels_list:
        testerrs = []
        trainerrs = []

        # split the train samples and labels to obtain different sample size
        X_trains = X_train[:n_sample, :]
        Y_trains = Y_train[:n_sample]

        # for loop changing feature size
        for feature in n_features_list:
            # split the training and testing samples to obtain different feature size
            xf = X_trains[:, :feature]
            xtf = X_test[:, :feature]

            # train the model
            w_l = leastsquare(xf, Y_trains)

            #compute the test error
            testerr = mean_squared_error(xtf @ w_l, Y_test)
            testerrs.append(testerr)
        testerrsa.append(testerrs)
    testerrsa_ = testerrsa_ + np.array(testerrsa)

testerrsa_ = testerrsa_ / 10
testerrsa1 = np.array(testerrsa)

# rescale the depth
testerrsa3 = sig(testerrsa1)

X, Y = np.meshgrid(n_features_list, n_sampels_list)

# use imshow function to draw the depth
plt.figure(1)
plt.imshow(testerrsa3,origin = 'lower')
plt.colorbar(label='Depth')

plt.xlabel('Features')
plt.ylabel('Samples')
plt.title('2D Depth Map of Test Error')

plt.savefig('DepthMap.jpg', dpi=300)
plt.show()
