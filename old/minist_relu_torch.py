import os
import struct
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import datetime
import torch
print(torch.cuda.is_available())
start = datetime.datetime.now()
data_path = r'D:\project\py\毕设\pythonProject\data\MNIST'

def Minst_Load(path, kind='train'):
    """Load MNIST data from `path`"""

    images_path = os.path.join(path, '{}-images.idx3-ubyte'.format(kind))
    labels_path = os.path.join(path, '{}-labels.idx1-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = torch.from_numpy(np.fromfile(lbpath, dtype=np.uint8))

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = torch.from_numpy(np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784))

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
noise_ratio = 0.2
n_noise = int(noise_ratio * len(Y))
idx = np.random.choice(len(Y), n_noise, replace=False)
#print(idx)
Y[idx] = - Y[idx] + 13

X, X_test, y, y_test = train_test_split(X, Y, test_size=0.8, random_state=52)

dim = X.shape[1]
N_relu = 2000
testerrs_all= torch.zeros(N_relu)
trainerrs_all= torch.zeros(N_relu)


Z = []
Zt = []
testerrs = []
accuracies = []
trainerrs = []
#with config_context(target_offload="gpu:0"):
#clf = SVC(kernel='linear',C=64)
for n_features in np.arange(0,N_relu,1):
    # Define the number of random ReLU features
    print(n_features)

    # Generate random ReLU features
    vi = np.random.normal(0, 0.05, size=dim)
    Zi = np.maximum(vi.dot(X.T),0)
    Zti = np.maximum(vi.dot(X_test.T), 0)
    Z.append(Zi)
    npZ = np.array(Z)
    Zt.append(Zti)
    npZi = np.array(Zt)

    npZ_1 = npZ / max(np.max(npZ),np.max(npZi))
    npZi_1 = npZi/ max(np.max(npZ),np.max(npZi))

    # Train SVM classifier

    ##clf.fit(npZ_1.T, y)

    # Test the model
    ##y_pre = clf.predict(npZi_1.T)
    ##testerr = np.mean((y_pre-y_test)**2)
    ##testerrs.append(testerr)

    ##y_tpre = clf.predict(npZ_1.T)
    ##trainerr = np.mean((y_tpre-y)**2)
    ##trainerrs.append(trainerr)

    if n_features % 100 == 0:
        end = datetime.datetime.now()
        print(end - start)
end = datetime.datetime.now()
print (end-start)

plt.plot(np.arange(0,N_relu,1),testerrs)
plt.plot(np.arange(0,N_relu,1),trainerrs)
plt.show()