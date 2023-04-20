import numpy as np
import matplotlib.pyplot as plt

# origin curve
x =  np.linspace(0, 1, 100) 
t =  np.sin(2 * np.pi * x)

font = {'family':'Times New Roman','size':20}
# sample function
def get_data(N):
    x_n = np.linspace(0,1,N)
    t_n = np.sin(2 * np.pi * x_n) + np.random.normal(scale=0.15, size=N) #add Gaussian Noise
    return x_n, t_n
# plot controller
def draw_ticks():
    plt.tick_params(labelsize=15)
    plt.xticks(np.linspace(0, 1, 2))
    plt.yticks(np.linspace(-1, 1, 3))
    plt.ylim(-1.5, 1.5)
    font = {'family':'Times New Roman','size':20}
    plt.xlabel('x', font)
    plt.ylabel('t',font, rotation='horizontal')
# sample
x_10, t_10 = get_data(10)

# plot section
plt.figure(1, figsize=(8,5))
plt.plot(x, t, 'g',linewidth=3)
plt.scatter(x_10, t_10, color='b', marker='o', edgecolors='b', s=100, linewidth=3, label="training data")
draw_ticks()
plt.title('Figure 1 : sample curve')
plt.savefig('1.png', dpi=400)
def regress(M, N, x, x_n, t_n, lamda=0):
    print("-----------------------M=%d, N=%d-------------------------" %(M,N))
    order = np.arange(M+1)
    order = order[:, np.newaxis]
    e = np.tile(order, [1,N])
    XT = np.power(x_n, e)
    X = np.transpose(XT)
    a = np.matmul(XT, X) + lamda*np.identity(M+1) #X.T * X
    b = np.matmul(XT, t_n) #X.T * T
    w = np.linalg.solve(a,b) #aW = b => (X.T * X) * W = X.T * T
    print("W:")
    print(w)
    e2 = np.tile(order, [1,x.shape[0]])
    XT2 = np.power(x, e2)
    p = np.matmul(w, XT2)
    return p
#M=0, N=10


p = regress(0, 10, x, x_10, t_10)
# plot section
plt.figure(2, figsize=(8,5))
plt.plot(x, t, 'g', x, p, 'r',linewidth=3)
plt.scatter(x_10, t_10, color='b', marker='o', edgecolors='b', s=100, linewidth=3)
draw_ticks()
plt.title('High bias', font)
plt.text(0.8, 0.9,'M = 0', font, style = 'italic')
plt.savefig('High bias.png', dpi=400)

#M=1, N=10
p = regress(1, 10, x, x_10, t_10)
# plot section
plt.figure(3, figsize=(8,5))
plt.plot(x, t, 'g', x, p, 'r',linewidth=3)
plt.scatter(x_10, t_10, color='b', marker='o', edgecolors='b', s=100, linewidth=3)
draw_ticks()
plt.title('Balance', font)
plt.text(0.8, 0.9,'M = 1', font, style = 'italic')
plt.savefig('Balance.png', dpi=400)

#M=3, N=10
p = regress(3, 10, x, x_10, t_10)
# plot section
plt.figure(4, figsize=(8,5))
plt.plot(x, t, 'g', x, p, 'r',linewidth=3)
plt.scatter(x_10, t_10, color='b', marker='o', edgecolors='b', s=100, linewidth=3)
draw_ticks()
plt.title(' M = 3, N = 10', font)
plt.text(0.8, 0.9,'M = 3', font, style = 'italic')
plt.savefig('4.png', dpi=400)
#M=9, N=10

p = regress(8, 10, x, x_10, t_10)
# plot section
plt.figure(5, figsize=(8,5))
plt.plot(x, t, 'g', x, p, 'r',linewidth=3)
plt.scatter(x_10, t_10, color='b', marker='o', edgecolors='b', s=100, linewidth=3)
draw_ticks()
plt.text(0.8, 0.9,'M = 8', font, style = 'italic')
plt.title(' M = 8, N = 10', font)
plt.savefig('5.png', dpi=400)


p = regress(100, 10, x, x_10, t_10)
plt.figure(6, figsize=(8,5))
plt.plot(x, t, 'g', x, p, 'r',linewidth=3)
plt.scatter(x_10, t_10, color='b', marker='o', edgecolors='b', s=100, linewidth=3)
draw_ticks()
plt.text(0.8, 0.9,'M = 100', font, style = 'italic')
plt.title('High variance', font)
plt.savefig('High variance.png', dpi=400)
