import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
N_relu = 6000

i = 61
testerrs = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'testerrs.csv',header=None))[0]
test01errs = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'test01errs.csv',header=None))[0]
n_supports = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'n_supports.csv',header=None))[0]
trainerrs = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'trainerrs.csv',header=None))[0]
ws = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'ws.csv',header=None))[0]


plt.figure(1)
plt.title("testerrs MSE")
plt.plot(np.arange(1,N_relu+1,1),testerrs)
#plt.savefig('F:\dataset\png\\'+ str(i) + 'testerrs.png')
plt.show()

plt.figure(2)
plt.title("testerrs01loss")
plt.plot(np.arange(1,N_relu+1,1),test01errs)
plt.show()

plt.figure(3)
plt.plot(np.arange(1,N_relu+1,1),n_supports)
plt.title("the number of support vectors")
plt.show()

plt.figure(4)
plt.plot(np.arange(1,N_relu+1,1),ws)
plt.title("ws")
plt.show()

plt.figure(5)
plt.plot(np.arange(1,N_relu+1,1),trainerrs/np.linalg.norm(trainerrs))
plt.title("trainerrs")
plt.show()

# plt.figure(6)
# plt.figure(figsize=(8,8))
# plt.subplot(311)
# plt.plot(np.arange(1,N_relu+1,1),test01errs)
# plt.plot(np.arange(1,N_relu+1,1),trainerrs/np.linalg.norm(trainerrs))
# plt.subplot(312)
# plt.plot(np.arange(1,N_relu+1,1),n_supports)
# plt.subplot(313)
# plt.plot(np.arange(1,N_relu+1,1),ws)
# plt.suptitle("ws")
# plt.show()
