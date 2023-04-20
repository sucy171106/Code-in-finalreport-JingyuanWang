import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# Generate a toy dataset
dim = 600
n_samples= 20
rng = np.random.RandomState(1)
# Uniformly distributed matrix
observed = rng.uniform(low=-1, high=1, size=(n_samples,dim))
labels = np.repeat([-1, 1], int((n_samples + 1) / 2))[:n_samples, None]  # drop last one if necessary
inputs = observed * labels # obtain the samples


def normalit(it):
    itsnorm = it/np.linalg.norm(it)
    return itsnorm


def SVM(x_train, y_train):
    # SVM
    # Define the coefficient of Hard margin SVM
    s_s = cp.Variable((x_train.shape[1], 1))
    # Two conditions
    objective = cp.Minimize(cp.norm(s_s) ** 2)
    constraints = [cp.multiply(y_train, x_train @ s_s) >= 1]
    # Define and solve the SVM Problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    s_s_value = s_s.value
    return s_s_value.ravel()


def softSVM(x_train, y_train):
    # Define the coefficients of Soft margin SVM
    w = cp.Variable((x_train.shape[1], 1))
    xi = cp.Variable((x_train.shape[0], 1, 1))
    C = 64

    # realise Soft SVM through CVXPY
    objective = cp.Minimize(0.5 * cp.norm(w) ** 2 + C * cp.sum(xi))
    constraints = [cp.multiply(y_train, x_train * w) >= 1 - xi, xi >= 0]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    prob.solve()
    w_value = w.value
    xi_value = xi.value
    return w_value.ravel(), xi_value.ravel()


def L2(x_train, y_train):
    n = x_train.shape[1]
    # define the coefficients
    s_s = cp.Variable((n, 1))
    # Two conditions
    objective = cp.Minimize(cp.norm(s_s))
    constraints = [y_train == x_train @ s_s]
    # Define and solve the Minimum L2 norm Problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    s_s_value = s_s.value
    return s_s_value.ravel()


def LS(x_train, y_train):
    w_l = np.linalg.pinv(x_train).dot(y_train).ravel()
    return w_l.ravel()


# GRADED FUNCTION: model_squreloss
def model_squreloss(x_train, y_train, num_iterations=2000, learning_rate=0.5):
    w_sl = np.zeros((x_train.shape[1], 1)) # GRADED FUNCTION: initialize_with_zeros
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate
        # the gradient of the weights
        dw_sl = -(x_train.T@ (y_train - x_train@w_sl ))/n_samples
        # update the weights
        w_sl = w_sl - learning_rate * dw_sl
        cost = (np.linalg.norm(y_train-x_train@w_sl )**2)/n_samples

        if i % 10000 == 0:
            costs.append(cost)
            # Print the cost every 1000 training examples
            print("Cost after iteration %i: %f" % (i, cost))
    return w_sl.ravel()






def LogisticLoss(w, X, y, lam):
    # Computes the cost function for all the training samples
    Xw = np.matmul(X,w)
    yT = y.reshape(-1,1)
    yXw = np.multiply(yT,Xw)
    f = np.mean(np.logaddexp(0, -yXw))  # cost

    # Computes the gradient
    gMul = np.exp(-yXw)/(1 + np.exp(-yXw))
    ymul = -1*yT*gMul
    g = np.matmul(ymul.reshape(1, -1), X)
    g = g.reshape(-1, 1)    # gradient
    return [f, g]


def model_logloss(x_train, y_train, num_iterations=20000, learning_rate=0.5):
    w_log = np.zeros((x_train.shape[1], 1))  # GRADED FUNCTION: initialize_with_zero
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate
        [cost, dw_log] = LogisticLoss(w_log, x_train, y_train, 1)
        w_log = w_log - learning_rate * dw_log
        # Print the cost every 1000 training examples
        if i % 100000 == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))
    return w_log.ravel()


w_svms = []
w_l2s = []
w_sls = []
w_logs = []
y = labels
for i in np.arange(n_samples, 200 + 1, 20):
    costs = []
    X = inputs[:, :i]
    w_svm = SVM(X, y)
    w_l2 = L2(X, y)
    w_sl = model_squreloss(X, y, num_iterations=20000, learning_rate=0.005)
    w_log = model_logloss(X, y, num_iterations=300000, learning_rate=0.0005)
    w_svms.append(normalit(w_svm))
    w_l2s.append(normalit(w_l2))
    w_sls.append(normalit(w_sl))
    w_logs.append(normalit(w_log))


w_svm600 = SVM(inputs, y)
w_l2600 = L2(inputs, y)
w_sl600 = model_squreloss(inputs, y, num_iterations=20000, learning_rate=0.005)
w_log600 = model_logloss(inputs, y, num_iterations=300000, learning_rate=0.0005)

print("The first 10 coefficients of Log loss using GD:")
print(normalit(w_log600)[0:10])
print("The first 10 coefficients of Hard margin SVM:")
print(normalit(w_svm600)[0:10])
print("The first 10 coefficients of Square Loss using GD:")
print(normalit(w_sl600)[0:10])
print("The first 10 coefficients of minimum L2 norm:")
print(normalit(w_l2600)[0:10])


def dis (x):
    dists =[]
    for i in x:
        dist = np.sum(abs(i))
        dists.append(dist)
    dists  = np.array(dists)

    return dists


def sigmoid(x):
    return 1/(1+np.exp(-x))

dim =200
plt.figure(1)
plt.plot( np.arange(n_samples,dim+1,20),np.log(dis(w_svms)), label='SVM')
plt.plot( np.arange(n_samples,dim+1,20),np.log(dis(w_logs)), label='LOG GD')
plt.axvline(120,ls='-.', label='SVP Threshold at 120',c='#FA7F6F')
plt.title("SVM & LOG GD")
plt.ylabel("Norm of the coefficients")
plt.xlabel("# Uniformly Random Features")
plt.legend()
plt.grid()
plt.savefig('one.jpg', dpi=300)


plt.figure(2)

plt.plot( np.arange(n_samples,dim+1,20),np.log(dis(w_l2s)), label='L2 norm')
plt.plot( np.arange(n_samples,dim+1,20),np.log(dis(w_sls)), label='SL GD')

# plt.axvline(120,ls='-.', label='SVP Threshold at 120',c='#FA7F6F')
plt.title("L2 norm & SL GD")
plt.ylabel("Norm of the coefficients")
plt.xlabel("# Uniformly Random Features")
plt.legend()
plt.grid()
plt.savefig('two.jpg', dpi=300)

plt.figure(3)
plt.plot( np.arange(n_samples,dim+1,20),np.log(dis(w_svms)), label='SVM')
plt.plot( np.arange(n_samples,dim+1,20),np.log(dis(w_l2s)), label='L2 norm')
plt.axvline(120,ls='-.', label='SVP Threshold at 120',c='#FA7F6F')
plt.title("SVM & L2 norm")
plt.ylabel("Norm of the coefficients")
plt.xlabel("# Uniformly Random Features")
plt.legend()
plt.grid()
plt.savefig('three.jpg', dpi=300)
plt.figure(4)
plt.plot( np.arange(n_samples,dim+1,20),np.log(dis(w_svms)), label='SVM')
plt.plot( np.arange(n_samples,dim+1,20),np.log(dis(w_l2s)), label='L2 norm')
plt.plot( np.arange(n_samples,dim+1,20),np.log(dis(w_sls)), label='SL GD')
plt.plot( np.arange(n_samples,dim+1,20),np.log(dis(w_logs)), label='LOG GD')
plt.axvline(120,ls='-.', label='SVP Threshold at 120',c='#FA7F6F')
plt.title("Norm of four models")
plt.ylabel("Norm of the coefficients")
plt.xlabel("# Uniformly Random Features")
plt.legend()
plt.grid()
plt.savefig('four.jpg', dpi=300)


plt.show()