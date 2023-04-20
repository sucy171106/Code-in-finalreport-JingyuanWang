import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#**** 3h running
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

# compute the distance
def dis (x):
    dists =[]
    for i in x:
        dist = np.sum(i)
        dists.append(dist)
    return dists


X, Y = Minst_Load(data_path)  # features: (60000, 784), labels: (60000,)
idx = (Y==5)+(Y==8) # get the index of 5 and 8
Y = Y[idx]   # change 8 to 1 and 5 to 0
X = X[idx]  # gte the matrix of the 5 and 8

# split 4000 samples
n_samples = 4000
X=X[:n_samples, :]/255
Y=(Y[:n_samples]-5)/3

costs=[]

# introduce noise into the labels
noise_ratio = 0.1
n_noise = int(noise_ratio * len(Y))
idx = np.random.choice(len(Y), n_noise, replace=False)
Y[idx] = - Y[idx] + 1   # noise is to Swap their values

# split the data and rescale x to 0-1
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=52)

dim = X.shape[1]
# the maximum relu feature is 1500
N_relu = 1500

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
k = np.arange(2, N_relu, 2)

testerrs = []
trainerrs = []
ws = []
v = []

# define the Logistic regression
clf = LogisticRegression(solver="saga", penalty="none", n_jobs=-1, max_iter=10000, random_state=0)

# generate the random relu matrix V
for n_features in np.arange(0, N_relu, 1):
    vi = np.random.uniform(-1, 1, size=dim)
    v.append(vi)
v = np.array(v)

for n_features in k:
    # Define the number of random ReLU features
    # Generate random ReLU features
    vn = v[:n_features, :]

    # increase the dimension Z is the train data Zt is the test data
    Z = np.maximum(vn @ X_train.T, 0)
    Zt = np.maximum(vn @ X_test.T, 0)

    # rescale the Z and Zt to [0,1]
    # npZ_1 is training samples, npZt_1 is the test samples
    npZ_1 = (2 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))) - 1
    npZt_1 = (2 * (Zt - np.min(Zt)) / (np.max(Zt) - np.min(Zt))) - 1

    # Train Logistic regression
    clf.fit(npZ_1.T, y_train)

    # the predicted value for test data
    y_pre = clf.predict(npZt_1.T)

    # the 01 test loss
    testerr = np.mean((y_pre - y_test) ** 2)
    testerrs.append(testerr)

    # the predicted value for training data
    y_tpre = clf.predict(npZ_1.T)

    # the 01 training loss
    trainerr = np.mean((y_tpre - y_train) ** 2)
    trainerrs.append(trainerr)

    # the norm of coefficients
    w = np.linalg.norm(clf.coef_)
    ws.append(w)

    # every 100 features output the running time
    if (n_features - 2) % 100 == 0:
        end = datetime.datetime.now()
        print(end - start)



# plot section
plt.figure(1)
thre=400
plt.grid()
plt.plot(k,testerrs , label='Test errors',c='#2878B5')
plt.plot(k,trainerrs , label='Train errors',c='#D76364')
plt.title("Test/Train Loss")
plt.ylabel("Test/Train errors")
plt.xlabel("# Random ReLU Features")
# plt.axvline(thre,ls='-.', label='Interpolation Threshold',c='#FA7F6F')
plt.xlim(0,N_relu)
plt.legend()
plt.legend(title='Noise = 0')
plt.savefig('TestTrainLoss_LOG_relu00.jpg', dpi=300)

plt.figure(2)
plt.plot(k,ws , label='Norm',c='#2878B5')
plt.title("Norm of ws")
plt.ylabel("ws")
plt.xlabel("# Random ReLU Features")
# plt.axvline(thre,ls='-.', label='Interpolation Threshold',c='#FA7F6F')
plt.xlim(0,N_relu)
plt.legend()
plt.grid()
plt.legend(title='Noise = 0')
plt.savefig('Norm_LOG_relu00.jpg', dpi=300)
plt.show()


plt.show()
