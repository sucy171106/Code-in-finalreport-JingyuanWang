
import struct
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# record the start time
start = datetime.datetime.now()

# the path of the data
data_path = r'F:\beproject\MNIST'

# load the minst
def Minst_Load(path, kind='train'):
    """Load MNIST data from `path`"""

    images_path = os.path.join(path, '{}-images.idx3-ubyte'.format(kind))
    labels_path = os.path.join(path, '{}-labels.idx1-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
    # images: n x m, n is number of samples, m is 784 pixels
    # labels: n

X, Y = Minst_Load(data_path)  # features: (60000, 784), labels: (60000,)
idx = (Y==5)+(Y==8) # get the index of 5 and 8
Y = Y[idx]  # change 8 to 1 and 5 to 0
X = X[idx]  # gte the matrix of the 5 and 8

# split 4000 samples
n_samples = 4000

# rescale the X, and change 8 to 1 and 5 to 0
X=X[:n_samples, :]/255
Y=(Y[:n_samples]-5)/3

costs=[]

# 01 loss
def zoloss(predict,ture)-> float:
    margin = 0.5
    prob_label = np.around(predict) #around
    N = ture.shape[0] #get the number of samples
    acc = np.sum(ture==prob_label) / N # compute the accuricy
    return  1-acc

# introduce noise into the labels
noise_ratio = 0.1
n_noise = int(noise_ratio * len(Y))
idx = np.random.choice(len(Y), n_noise, replace=False)
Y[idx] = - Y[idx] + 1   # noise is to Swap their values

# split the data and rescale x to 0-1
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=52)

dim = X.shape[1]
# the maximum relu feature is 1500
N_relu = 2000

# restore the results
testerrs_all= np.zeros(N_relu)
trainerrs_all= np.zeros(N_relu)

y_train = y_train.ravel()
y_test = y_test.ravel()

# pre view the data
plt.figure(1, figsize=(20,5))
for i in range(4):
    idx = np.random.choice(range(6))
    plt.subplot(int('14'+str(i+1)))
    plt.imshow(X_train[idx,:].reshape((28,28)))
    plt.title('label is %d'%y_train[idx])
plt.show()

# sweep the relu feature
k = np.arange(2, N_relu, 20)

# restore the results
testerrsa = np.zeros(k.shape[0])
trainerrsa = np.zeros(k.shape[0])
w_s = np.zeros(k.shape[0])
v = []

# define the linear regression
clf = LinearRegression(n_jobs=-1)

# generate the random relu matrix V
for n_features in np.arange(0, N_relu, 1):
    vi = np.random.uniform(-1, 1, size=dim)
    v.append(vi)
v = np.array(v)

# run 10 times
times = 10
for i in range(0, times, 1):
    testerrs = []
    trainerrs = []
    ws = []

    for n_features in k:
        # Define the number of random ReLU features
        # Generate random ReLU features
        vn = v[:n_features, :]

        # increase the dimension Z is the train data Zt is the test data
        Z = np.maximum(vn @ X_train.T, 0)
        Zt = np.maximum(vn @ X_test.T, 0)

        # rescale the Z and Zt
        npZ_1 = (2 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))) - 1
        npZt_1 = (2 * (Zt - np.min(Zt)) / (np.max(Zt) - np.min(Zt))) - 1

        # Train linear regression
        clf.fit(npZ_1.T, y_train)

        # the predicted value for test data
        y_pre = clf.predict(npZt_1.T)

        # the 01 test loss
        testerr = zoloss(y_pre, y_test)
        testerrs.append(testerr)

        # the predicted value for training data
        y_tpre = clf.predict(npZ_1.T)

        # the 01 training loss
        trainerr = zoloss(y_tpre, y_train)
        trainerrs.append(trainerr)

        # the norm of coefficients
        w = np.linalg.norm(clf.coef_)
        ws.append(w)

        # every 100 features output the running time
        if (n_features - 2) % 100 == 0:
            end = datetime.datetime.now()
            print(end - start)

    testerrsa = testerrsa + np.array(testerrs)
    trainerrsa = trainerrsa + np.array(trainerrs)
    w_s = w_s + np.array(ws)


# plot section
plt.plot(k/800,trainerrsa/times, label='Train Risk', c='#D76364')
plt.plot(k/800,testerrsa/times, label='Test Risk',c='#2878B5')
plt.axvline(1,ls='-.', label='Interpolation Threshold', c='#FA7F6F')##EF7A6D#D76364
plt.grid()

plt.title("Test/Train 01Loss of LS vs Ratio (Dimensions/Samples)")
plt.ylabel("Test/Train 01Loss Risk")
plt.xlabel("# Dimensions/Samples")

plt.ylim(0,1)
plt.xlim(0,2.5)

plt.legend(title='Noise = 0.1')
plt.savefig('TestTraiRisk_MNIST.jpg', dpi=300)

plt.show()

plt.plot(k/800,w_s/times, label='Norm', c='#2878B5')
plt.title("Norm of w vs  Ratio (Dimensions/Samples)")
plt.ylabel("Norm")
plt.xlabel("# Dimensions/Samples")
plt.ylim(0,150)
plt.xlim(0,2.5)
plt.axvline(1,ls='-.', label='Interpolation Threshold',c='#FA7F6F')

plt.text(1, w_s[40]/times, "MAX", horizontalalignment='right')
plt.scatter(1, w_s[40]/times)

plt.legend(title='Noise = 0.1')
plt.grid()
plt.savefig('Norm_MNIST.jpg', dpi=300)

plt.show()
