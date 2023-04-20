import os
import struct
import numpy as np
from sklearn.metrics import zero_one_loss
from sklearn.svm import SVC
import datetime
from sklearn.model_selection import train_test_split
import csv

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

# the VC dimension. actually this is not mentioned in the paper
def VCdimensinon(x,le):
    return min(le,np.array(x)**2)+1


X, Y = Minst_Load(data_path)  # features: (60000, 784), labels: (60000,)
idx = (Y == 5)+(Y == 8) # get the index of 5 and 8
Y = (Y[idx] - 5) / 3    # change 8 to 1 and 5 to 0
X = X[idx]  # gte the matrix of the 5 and 8

# split 4000 samples
n_samples = 4000
X=X[:n_samples, :]
Y=Y[:n_samples]

# introduce noise into the labels
# noise_ratio = 0.0
# noise_ratio = 0.1
noise_ratio = 0.2
n_noise = int(noise_ratio * len(Y))
idx = np.random.choice(len(Y), n_noise, replace=False)
Y[idx] = - Y[idx] + 1   # noise is to Swap their values

# split the data and rescale x to 0-1
X_train, X_test, y_train, y_test = train_test_split(X/255, Y, test_size=0.8, random_state=52)

dim = X.shape[1]
# the maximum relu feature is 1500
N_relu = 1500

# define the function of SVM
clf = SVC(kernel='linear',C=64)

k = np.arange(2,N_relu,2)

# restore the results
l = len(k)
testerrs_all= np.zeros(l)
trainerrs_all= np.zeros(l)
ws_all = np.zeros(l)
n_supports_all = np.zeros(l)

# define the lists
Z = []
Zt = []
testerrs = []
accuracies = []
trainerrs = []
ws = []
n_supports = []
v=[]
VCs=[]

# generate the random relu matrix V
for n_features in np.arange(0,N_relu,1):
    vi = np.random.uniform(-1, 1, size=dim)
    v.append(vi)
v = np.array(v)

# sweep the features
for n_features in k:
    # Define the number of random ReLU features

    # Generate random ReLU features
    vn = v[:n_features,:]

    # increase the dimension Z is the train data Zt is the test data
    Z = np.maximum(vn@X_train.T,0)
    Zt = np.maximum(vn@X_test.T, 0)

    # rescale the Z and Zt
    npZ_1 = (2 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))) - 1
    npZt_1 = (2 * (Zt - np.min(Zt)) / (np.max(Zt) - np.min(Zt))) - 1

    # Train  SVM classifier
    clf.fit(npZ_1.T, y_train)

    # the norm of coefficients
    w =np.linalg.norm(clf.coef_)

    # the estimated VC dimension
    VC = VCdimensinon(w,n_features)
    VCs.append(VC)

    # every 100 features output the running time
    if n_features % 100 == 0:
        end = datetime.datetime.now()
        print(end - start)

end = datetime.datetime.now()
print(end-start)

# export the test error in CSV file
with open('VCs.csv', 'w', encoding='utf-8', newline='') as file_obj:
    # create the object
    writer = csv.writer(file_obj)
    writer.writerow((VCs))

# export the number of support vectors in CSV file

