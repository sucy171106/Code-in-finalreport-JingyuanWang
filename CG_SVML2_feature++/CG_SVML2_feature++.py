import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# Generate a toy dataset feature size is 100 sample size is 20
dim = 100
sample = 20

rng = np.random.RandomState(1)
observed = rng.uniform(low=-1, high=1, size=(sample, dim)) # uniform distributed

labels = np.repeat([-1, 1], int((sample + 1) / 2))[:sample, None]  # drop last one if necessary
inputs = observed * labels

X = observed
y = labels

# feature changes from the sample size to dim, overparameterized
n_features_list = np.arange(sample, dim + 1, 2)
distancesa = np.zeros(n_features_list.shape[0])

for i in range(50):
    distances = []
    print(i)

    for n_features in n_features_list:
        # split the feature
        x = X[:, :n_features]
        n = x.shape[1]

        # the minimum L2 norm solution
        s_LR = cp.Variable((n, 1))

        objective1 = cp.Minimize(cp.norm(s_LR))
        constraints1 = [y == x @ s_LR]

        prob1 = cp.Problem(objective1, constraints1)
        prob1.solve()

        # restore the value
        s_LR_value = s_LR.value
        w_l = s_LR_value

        # the SVM model
        s_s = cp.Variable((n, 1))

        objective = cp.Minimize(cp.norm(s_s) ** 2)
        constraints = [cp.multiply(y, x @ s_s) >= 1]

        prob = cp.Problem(objective, constraints)

        result = prob.solve()
        s_s_value = s_s.value
        w_s = s_s_value

        # computing the distance
        distance = np.linalg.norm(w_s - w_l)
        distances.append(distance)
    distancesa = distancesa + np.array(distances)


# plot section
plt.plot(n_features_list, distancesa/40,c='#2878B5')
plt.xlabel("# Dimensions")
plt.ylabel("Distances")
plt.title("Distances between SVM and L2 vs Dimensions")
plt.grid()

plt.savefig('SVM_L2_Iterations.jpg', dpi=300)
plt.show()