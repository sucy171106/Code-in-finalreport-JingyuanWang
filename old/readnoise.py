import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
N_relu = 2000

i = 0

# plt.figure(1)
# plt.title("testerrs MSE")
# plt.plot(np.arange(1,N_relu+1,1),testerrs)
# #plt.savefig('F:\dataset\png\\'+ str(i) + 'testerrs.png')
# plt.show()
#
# plt.figure(2)
# plt.title("testerrs01loss")
# plt.plot(np.arange(1,N_relu+1,1),test01errs)
# plt.show()
#
# plt.figure(3)
# plt.plot(np.arange(1,N_relu+1,1),n_supports)
# plt.title("the number of support vectors")
# plt.show()
#
# plt.figure(4)
# plt.plot(np.arange(1,N_relu+1,1),ws)
# plt.title("ws")
# plt.show()
#
# plt.figure(5)
# plt.plot(np.arange(1,N_relu+1,1),trainerrs/np.linalg.norm(trainerrs))
# plt.title("trainerrs")
# plt.show()


plt.figure(figsize=(6,8))
plt.subplot(411)
i = 0
test01errs = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'testerrs.csv',header=None))[0]
plt.plot(np.arange(1,N_relu+1,1),test01errs)
i = 1
test01errs = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'testerrs.csv',header=None))[0]
plt.plot(np.arange(1,N_relu+1,1),test01errs)
i = 2
test01errs = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'testerrs.csv',header=None))[0]
plt.plot(np.arange(1,N_relu+1,1),test01errs)
plt.subplot(412)
i = 0
trainerrs = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'trainerrs.csv',header=None))[0]
plt.plot(np.arange(1,N_relu+1,1),trainerrs)
i = 1
trainerrs = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'trainerrs.csv',header=None))[0]
plt.plot(np.arange(1,N_relu+1,1),trainerrs)
i = 2
trainerrs = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'trainerrs.csv',header=None))[0]
plt.plot(np.arange(1,N_relu+1,1),trainerrs)

plt.subplot(413)
i = 0
n_supports = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'n_supports.csv',header=None))[0]
plt.plot(np.arange(1,N_relu+1,1),n_supports)
i = 1
n_supports = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'n_supports.csv',header=None))[0]
plt.plot(np.arange(1,N_relu+1,1),n_supports)
i = 2
n_supports = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'n_supports.csv',header=None))[0]
plt.plot(np.arange(1,N_relu+1,1),n_supports)
plt.subplot(414)
i = 0
ws = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'ws.csv',header=None))[0]
plt.plot(np.arange(1,N_relu+1,1),ws)
i = 1
ws = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'ws.csv',header=None))[0]
plt.plot(np.arange(1,N_relu+1,1),ws)
i = 2
ws = np.array(pd.read_csv('F:\dataset\q'+ str(i) + 'n\q'+ str(i) + 'ws.csv',header=None))[0]
plt.plot(np.arange(1,N_relu+1,1),ws)
plt.suptitle("ws")
plt.show()
