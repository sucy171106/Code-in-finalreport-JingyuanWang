import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Generate a toy dataset
def data(nsamples, ndim):
    X, Y = datasets.make_regression(n_samples=nsamples, n_features=ndim,
                                    n_informative=ndim, noise=0, random_state=None)
    X = X / np.std(X)  # rescale, make its standard deviation to zero
    Y = Y / np.std(Y)  # rescale, make its standard deviation to zero
    Y = Y + np.random.normal(0, 0.1, Y.shape)  # add noise to the label

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  # split the data
    return X_train, X_test, Y_train, Y_test


# least squares model function
def leastsquare(x_train, y_train):
    # Least squares
    w_l = np.linalg.pinv(x_train).dot(y_train).ravel()
    return w_l.ravel()


# the number of the samples is 400, the dimension size is 100
n_samples = 400
dim = 100
n_sampels_list = np.arange(1, 200 + 1, 1)

# restore the results
testerrsa = np.zeros(n_sampels_list.shape[0])
trainerrsa = np.zeros(n_sampels_list.shape[0])
w_lnsa = np.zeros(n_sampels_list.shape[0])

#  run 40 times
times = 40
for i in range(0, times, 1):
    testerrs = []
    trainerrs = []
    w_lns = []

    # generate the data
    X_train, xt, Y_train, yt = data(n_samples, dim)

    # for loop changing sample size
    for sampel in n_sampels_list:
        # split the train samples and labels to obtain different sample size
        x = X_train[:sampel, :]
        y = Y_train[:sampel]
        w_l = leastsquare(x, y)

        # compute the test error
        testerr = mean_squared_error(xt @ w_l, yt)
        testerrs.append(testerr)

        # compute the test error
        trainerr = mean_squared_error(x @ w_l, y)
        trainerrs.append(trainerr)

        # normalize the weights
        w_ln = np.linalg.norm(w_l)
        w_lns.append(w_ln)

    testerrsa = testerrsa + np.array(testerrs)
    trainerrsa = trainerrsa + np.array(trainerrs)
    w_lnsa = w_lnsa + np.array(w_lns)


# plot section
plt.plot(n_sampels_list, testerrsa / times, label='Test Risk', c='#2878B5')
plt.plot(n_sampels_list, trainerrsa / times, linewidth=2, label='Train Risk', c='#D76364')
plt.axvline(n_samples / 4, ls='-.', label='Interpolation Threshold', c='#FA7F6F')
plt.grid()

plt.title("Test/Train Risk vs Samples (100 Dimensions)")
plt.ylabel("Test/Train MSE Risk")
plt.xlabel("# Samples")

plt.ylim(0, 2)
plt.xlim(0, 200)

plt.legend()
plt.savefig('TestTraiRisk_Samples.jpg', dpi=300)
plt.show()
plt.plot(n_sampels_list, w_lnsa / times, label='Norm of w', c='#2878B5')
plt.title("Norm of w vs Samples (100 Dimensions)")

plt.ylabel("Norm")
plt.xlabel("# Samples")

plt.ylim(0, 5)
plt.xlim(0, 200)

plt.axvline(n_samples / 4, ls='-.', label='Interpolation Threshold', c='#FA7F6F')

plt.text(100, w_lnsa[99] / 40, "MAX", horizontalalignment='right')
plt.scatter(100, w_lnsa[99] / 40)

plt.legend()
plt.grid()
plt.savefig('Norm_Samples.jpg', dpi=300)
plt.show()
