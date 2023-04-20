import os
import struct
import numpy as np
from sklearn.metrics import zero_one_loss
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
Y = (Y[idx]-5)/3
X = X[idx]

n_samples = 4000
X=X[:n_samples, :]
Y=Y[:n_samples]

# print(X[1,:])
# introduce noise into the labels
noise_ratio = 0
n_noise = int(noise_ratio * len(Y))
idx = np.random.choice(len(Y), n_noise, replace=False)
Y[idx] = - Y[idx] + 1

X_train, X_test, y_train, y_test = train_test_split(X/255, Y, test_size=0.8, random_state=52)

dim = X.shape[1]
N_relu = 1500
testerrs_all= np.zeros(N_relu)
trainerrs_all= np.zeros(N_relu)



#with config_context(target_offload="gpu:0"):
clf = SVC(kernel='linear',C=64)
def VCdimensinon(x,le):
    return min(le,np.array(x)**2)+1


k = np.arange(2, N_relu, 8)
Z = []
Zt = []
testerrs = []
accuracies = []
trainerrs = []
ws = []
n_supports = []
v = []
VCs = []
for n_features in np.arange(0, N_relu, 1):
    # vi = np.random.normal(0, 0.05, size=dim)
    vi = np.random.uniform(-1, 1, size=dim)
    v.append(vi)
v = np.array(v)
print(v.shape)

for n_features in k:
    # Define the number of random ReLU features
    #     print(n_features)

    # Generate random ReLU features
    vn = v[:n_features, :]
    # Zi = np.maximum(vn.dot(X.T),0)
    Z = np.maximum(vn @ X_train.T, 0)
    # print(vi.shape)
    #     print(Z.shape)
    # print(X.T.shape)
    # print(Zi.shape)
    # Zti = np.maximum(vn.dot(X_test.T), 0)
    Zt = np.maximum(vn @ X_test.T, 0)
    #     Z.append(Zi)
    #     npZ = np.array(Z)
    #     Zt.append(Zti)
    #     npZt = np.array(Zt)
    # print(npZ.shape)
    # print(np.max(Z))
    # print(np.min(Z))
    # npZ_1 = npZ / np.linalg.norm(npZ)

    npZ_1 = (2 * (Z - np.min(Z)) / (np.max(Z) - np.min(Z))) - 1
    # npZt_1 = npZt/ np.linalg.norm(npZt)
    npZt_1 = (2 * (Zt - np.min(Zt)) / (np.max(Zt) - np.min(Zt))) - 1

    # Train  SVM classifier

    clf.fit(npZ_1.T, y_train)

    # Test the model
    y_pre = clf.predict(npZt_1.T)
    testerr = zero_one_loss(y_pre, y_test)
    testerrs.append(testerr)

    y_tpre = clf.predict(npZ_1.T)
    trainerr = zero_one_loss(y_tpre, y_train)
    trainerrs.append(trainerr)

    n_support = np.mean(clf.n_support_)
    # print(n_support)

    n_supports.append(n_support)
    # print(n_supports)
    w = np.linalg.norm(clf.coef_)
    ws.append(w)
    VC = VCdimensinon(w, n_features)
    VCs.append(VC)
    if n_features % 100 == 0:
        end = datetime.datetime.now()
        print(end - start)

end = datetime.datetime.now()
print(end - start)


thre = 100
plt.figure(1)
plt.grid()
plt.plot(k,testerrs , label='Test errors',c='#2878B5')
plt.plot(k,trainerrs , label='Train errors',c='#D76364')
plt.title("Test/Train 01Loss")
plt.ylabel("Test/Train errors")
plt.xlabel("# Random ReLU Features")
plt.axvline(thre,ls='-.', label='Interpolation Threshold',c='#FA7F6F')
plt.xlim(0,N_relu)
plt.legend()
plt.legend(title='Noise = 0')
plt.savefig('TestTrain01Loss_SVM_relu.jpg', dpi=300)

plt.figure(2)
plt.plot(k,n_supports , label='Support vectors',c='#2878B5')
plt.title("The number of support vectors")
plt.ylabel("Support vectors")
plt.xlabel("# Random ReLU Features")
plt.axvline(thre,ls='-.', label='Interpolation Threshold',c='#FA7F6F')
plt.xlim(0,N_relu)
plt.legend()
plt.grid()
plt.legend(title='Noise = 0')
plt.savefig('supportvectors_SVM_relu.jpg', dpi=300)


plt.figure(4)
plt.plot(k,ws , label='Norm',c='#2878B5')
plt.title("Norm of ws")
plt.ylabel("ws")
plt.xlabel("# Random ReLU Features")
plt.axvline(thre,ls='-.', label='Interpolation Threshold',c='#FA7F6F')
plt.xlim(0,N_relu)
plt.legend()
plt.grid()
plt.legend(title='Noise = 0')
plt.savefig('Norm_SVM_relu.jpg', dpi=300)
plt.show()

# plt.figure(5)


# plt.plot(k,VCs)
# plt.title("VC dimension")
# plt.ylabel("VC dimension")
# plt.xlabel("# Random ReLU Features")
# plt.axvline(thre,ls='-.', label='Interpolation Threshold',c='#FA7F6F')
# plt.xlim(0,N_relu)
# plt.legend()

