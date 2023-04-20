import os
import struct
import numpy as np

from sklearn.svm import SVC
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split

start = datetime.datetime.now()
data_path = r'F:\beproject\MNIST'

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
idx = (Y==5)+(Y==8)
Y = Y[idx]
X = X[idx]

n_samples = 4000
X=X[:n_samples, :]
Y=Y[:n_samples]

# introduce noise into the labels
noise_ratio = 0.1
n_noise = int(noise_ratio * len(Y))
idx = np.random.choice(len(Y), n_noise, replace=False)
Y[idx] = - Y[idx] + 13

X_train, X_test, y_train, y_test = train_test_split(X/255, Y, test_size=0.8, random_state=52)

dim = X.shape[1]
N_relu = 1000
testerrs_all= np.zeros(N_relu)
trainerrs_all= np.zeros(N_relu)



#with config_context(target_offload="gpu:0"):
clf = SVC(kernel='linear',C=64)
k = np.arange(2, N_relu, 15)
Z = []
Zt = []
testerrs = []
accuracies = []
trainerrs = []
ws = []
n_supports = []
v = []
for n_features in np.arange(0, N_relu, 1):
    # vi = np.random.normal(0, 0.05, size=dim)
    vi = np.random.uniform(-1, 1, size=dim)
    v.append(vi)
v = np.array(v)
print(v.shape)

for n_features in k:
    # Define the number of random ReLU features
    # print(n_features)

    # Generate random ReLU features
    vn = v[:n_features, :]
    Z = np.maximum(vn @ X_train.T, 0)
    Zt = np.maximum(vn @ X_test.T, 0)

    npZ_1 = (2 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))) - 1
    npZt_1 = (2 * (Zt - np.min(Zt)) / (np.max(Zt) - np.min(Zt))) - 1

    clf.fit(npZ_1.T, y_train)
    y_pre = clf.predict(npZt_1.T)
    testerr = np.mean((y_pre - y_test) ** 2)
    testerrs.append(testerr)

    y_tpre = clf.predict(npZ_1.T)
    trainerr = np.mean((y_tpre - y_train) ** 2)
    trainerrs.append(trainerr)

    n_support = np.mean(clf.n_support_)
    n_supports.append(n_support)

    w = np.linalg.norm(clf.coef_)
    ws.append(w)

    if n_features % 100 == 0:
        end = datetime.datetime.now()
        print(end - start)

end = datetime.datetime.now()
print(end - start)

plt.figure(1)
plt.subplot(221)
plt.plot(k,testerrs)
plt.suptitle("testerrs MSE")
plt.subplot(222)

plt.plot(k,n_supports)
plt.suptitle("the number of support vectors")
plt.subplot(223)
plt.plot(k,trainerrs)
plt.suptitle("trainerrs MSE")
plt.subplot(224)
plt.plot(k,ws)
plt.suptitle("ws")
plt.show()
