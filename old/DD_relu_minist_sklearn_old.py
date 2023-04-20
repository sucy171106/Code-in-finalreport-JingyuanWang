import os
import struct
import numpy as np

from sklearn.svm import SVC
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss

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

X, X_test, y, y_test = train_test_split(X, Y, test_size=0.8, random_state=52)

dim = X.shape[1]
N_relu = 2000
testerrs_all= np.zeros(N_relu)
trainerrs_all= np.zeros(N_relu)


Z = []
Zt = []
testerrs = []
accuracies = []

test01errs = []

trainerrs = []
n_supports = []
ws = []
#with config_context(target_offload="gpu:0"):
clf = SVC(kernel='linear',C=64)
for n_features in np.arange(1,N_relu+1,1):
    # Define the number of random ReLU features
    #print(n_features)

    # Generate random ReLU features
    vi = np.random.normal(0, 0.05, size=dim)
    Zi = np.maximum(vi.dot(X.T),0)
    #print(vi.shape)
    #print(X.T.shape)
    #print(Zi.shape)
    Zti = np.maximum(vi.dot(X_test.T), 0)
    Z.append(Zi)
    npZ = np.array(Z)
    Zt.append(Zti)
    npZt = np.array(Zt)

    npZ_1 = npZ / max(np.max(npZ),np.max(npZt))
    npZt_1 = npZt/ max(np.max(npZ),np.max(npZt))

    # Train  SVM classifier

    clf.fit(npZ_1.T, y)

    # Test the model
    y_pre = clf.predict(npZt_1.T)
    testerr = np.mean((y_pre-y_test)**2)
    testerrs.append(testerr)
    test01err = zero_one_loss(y_pre,y_test)
    test01errs.append(test01err)

    y_tpre = clf.predict(npZ_1.T)
    trainerr = np.mean((y_tpre-y)**2)
    trainerrs.append(trainerr)

    n_support = np.mean(clf.n_support_)
    #print(n_support)

    n_supports.append(n_support)
    #print(n_supports)
    w =np.linalg.norm(clf.coef_)
    ws.append(w)

    if n_features % 100 == 0:
        end = datetime.datetime.now()
        print(end - start)

end = datetime.datetime.now()
print (end-start)

import csv

with open('1testerrs4000.csv', 'w', encoding='utf-8', newline='') as file_obj:
    # 创建对象
    writer = csv.writer(file_obj)
    writer.writerow((testerrs))

with open('1test01errs4000.csv', 'w', encoding='utf-8', newline='') as file_obj:
    # 创建对象
    writer = csv.writer(file_obj)

    writer.writerow((test01errs))

with open('1n_supports4000.csv', 'w', encoding='utf-8', newline='') as file_obj:
    # 创建对象
    writer = csv.writer(file_obj)

    writer.writerow((n_supports))

with open('1trainerrs4000.csv', 'w', encoding='utf-8', newline='') as file_obj:
    # 创建对象
    writer = csv.writer(file_obj)

    writer.writerow((testerrs))

with open('1ws4000.csv', 'w', encoding='utf-8', newline='') as file_obj:
    # 创建对象
    writer = csv.writer(file_obj)
    writer.writerow((ws))