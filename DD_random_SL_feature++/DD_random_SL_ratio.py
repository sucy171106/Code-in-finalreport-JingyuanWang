import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Generate a toy dataset
def data(nsamples, ndim):
    X, Y = datasets.make_regression(n_samples=nsamples, n_features=ndim,
                                    n_informative=ndim, noise=0, random_state=None)
    X = X / np.std(X) # rescale, make its standard deviation to zero
    Y = Y / np.std(Y) # rescale, make its standard deviation to zero
    Y = Y + np.random.normal(0, 0.1, Y.shape)  # add noise to the label

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)  # split the data
    return X_train, X_test, Y_train, Y_test

# least squares model function
def leastsquare(x_train, y_train):
    # LS
    w_l = np.linalg.pinv(x_train).dot(y_train).ravel()
    return w_l.ravel()


# the number of the samples is 125, the dimension size is 300
n_samples = 125
dim = 300

# feature size changes from 1 to 200
n_features_list = np.arange(0, dim+1, 1)

# restore the results from different times
testerrsa = np.zeros(n_features_list.shape[0])
trainerrsa = np.zeros(n_features_list.shape[0])
w_lnsa = np.zeros(n_features_list.shape[0])

for i in range(0,40,1):
    testerrs = []
    trainerrs =[]
    w_lns = []
    X_train, X_test, Y_train, Y_test = data(n_samples, dim)

    # for loop changing feature size
    for feature in n_features_list:
        # split the training and testing samples to obtain different feature size
        x=X_train[:,:feature]
        xt=X_test[:,:feature]

        # train the model
        w_l = leastsquare(x, Y_train)

        # compute the test error
        testerr = mean_squared_error(xt@w_l , Y_test)
        testerrs.append(testerr)

        # compute the train error
        trainerr = mean_squared_error(x@w_l , Y_train)
        trainerrs.append(trainerr)

        # normalize the weights
        w_ln = np.linalg.norm(w_l)
        w_lns.append(w_ln)

    testerrsa = testerrsa+np.array(testerrs)
    trainerrsa = trainerrsa + np.array(trainerrs)
    w_lnsa = w_lnsa + np.array(w_lns)

# plot section
plt.plot(n_features_list/ 100 , testerrsa/40 , label='Test Risk',c='#2878B5')
plt.plot(n_features_list / 100, trainerrsa/40, linewidth=1, label='Train Risk', c='#D76364')#FA7F6F
plt.axvline(1,ls='-.', label='Interpolation Threshold', c='#FA7F6F')##EF7A6D#D76364

plt.grid()

plt.title("Test/Train MSE Risk vs Ratio (Dimensions/Samples)")
plt.ylabel("Test/Train MSE Risk")
plt.xlabel("# Dimensions/Samples")

plt.ylim(0,10)
plt.xlim(0,3)
plt.yticks(range(0,11))
plt.legend()
plt.savefig('TestTraiRisk_Ratio.jpg', dpi=300)

plt.show()

plt.plot(n_features_list/100 , w_lnsa/40 , label='Norm of w',c='#2878B5')
plt.title("Norm of w vs  Ratio (Dimensions/Samples)")

plt.ylabel("Norm")
plt.xlabel("# Dimensions/Samples")

plt.ylim(0,50)
plt.xlim(0,3)

plt.axvline(1,ls='-.', label='Interpolation Threshold',c='#FA7F6F')
plt.text(1, w_lnsa[100]/40, "MAX")
plt.scatter(1, w_lnsa[100]/40)
plt.legend()
plt.grid()
plt.savefig('Norm_Features.jpg', dpi=300)
plt.show()