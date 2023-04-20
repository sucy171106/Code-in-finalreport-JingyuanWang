import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import cvxpy as cp
import scipy as sc

# Generate a toy dataset
n_samples = 20
dim = 60000

X_train, Y = datasets.make_blobs(n_samples=n_samples, n_features=dim, centers=2, cluster_std=[2.0, 2.0],
                                 random_state=42)
Y = Y * 2 - 1
Y_train = np.reshape(Y, (len(Y), 1))


ss = sc.sparse.linalg.svds(X_train)

s0 = np.max(ss[0])
s1 = np.max(ss[1])
s2 = np.max(ss[2])

lr = 1 / (s1 ** 2)
print(1 / (s1 ** 2))


def SVM(x_train, y_train):
    # SVM
    s_s = cp.Variable((x_train.shape[1], 1))

    objective = cp.Minimize(cp.norm(s_s) ** 2)
    constraints = [cp.multiply(y_train, x_train @ s_s) >= 1]
    prob = cp.Problem(objective, constraints)

    prob.solve(solver=cp.SCS)
    s_s_value = s_s.value

    return s_s_value


def LogisticLoss(w, X, y, lam):
    # Computes the cost function for all the training samples
    m = X.shape[0]
    Xw = np.matmul(X, w)
    yT = y.reshape(-1, 1)
    yXw = np.multiply(yT, Xw)

    f = np.sum(np.logaddexp(0, -yXw))
    gMul = np.exp(-yXw) / (1 + np.exp(-yXw))
    ymul = -1 * yT * gMul

    g = np.matmul(ymul.reshape(1, -1), X)

    g = g.reshape(-1, 1)
    return [f, g]


def model_logloss(x_train, y_train, c, num_iterations, learning_rate):
    w_log = np.zeros((x_train.shape[1], 1))  # GRADED FUNCTION: initialize_with_zeros
    distances = []
    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate

        [cost, dw_log] = LogisticLoss(w_log, x_train, y_train, 1)

        w_log = w_log - learning_rate * dw_log

        if i % 1000 == 0:
            distance = np.linalg.norm(w_log / np.linalg.norm(w_log) - c / np.linalg.norm(c))
            distances.append(distance)
            print("Cost after iteration %i: %f" % (i, cost))
            # Print the cost every 1000 training examples
    return distances


def train(lr):
    # train with SVM model
    w_s_in = SVM(X_train, Y_train)

    distances = model_logloss(X_train, Y_train, w_s_in, 500000, lr)

    return distances

n_features_list = np.arange(10, dim+1, 100)
costs =[]

w_log_norms = []
w_s_norms = []
# Calculate the maximum singular value
ss = sc.sparse.linalg.svds(X_train)
s1 = np.max(ss[1])



distance2 = train(1/(s1**2)*0.1)
print(len(distance2))

plt.plot(range(0,500000,1000),distance2,c='#2878B5')
plt.scatter(range(0,500000,1000),distance2,c='#2878B5',s=20)
plt.xlabel("# Iterations in log scale")
plt.ylabel("Distances")
plt.title("Distances between Log and SVM vs Iterations")

# plt.legend()
plt.grid()
plt.xscale('log')
plt.savefig('SL_L2_Iterationslog.jpg', dpi=300)
plt.show()


plt.plot(range(0,500000,1000),distance2,c='#2878B5')
plt.scatter(range(0,500000,1000),distance2,c='#2878B5',s=20)
plt.xlabel("# Iterations")
plt.ylabel("Distances")
plt.title("Distances between Log and SVM vs Iterations")

# plt.legend()
plt.grid()
# plt.xscale('log')
plt.savefig('SL_L2_Iterations1.jpg', dpi=300)
plt.show()