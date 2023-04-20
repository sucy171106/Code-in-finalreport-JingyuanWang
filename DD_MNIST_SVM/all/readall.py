import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
N_relu = 1500
i=7
# if i == 9:
#     thre = 140
# if i ==8:
#     thre = 330
# if i == 7:
#     thre = 410
testerrs02 = np.array(pd.read_csv('q'+ str(i) + 'testerrs.csv',header=None))[0]
n_supports02 = np.array(pd.read_csv('q'+ str(i) + 'n_supports.csv',header=None))[0]
trainerrs02 = np.array(pd.read_csv('q'+ str(i) + 'trainerrs.csv',header=None))[0]
ws02 = np.array(pd.read_csv('q'+ str(i) + 'ws.csv',header=None))[0]
# k = np.arange(2,N_relu,20)
k = np.arange(2,N_relu,2)

i=8
testerrs01 = np.array(pd.read_csv('q'+ str(i) + 'testerrs.csv',header=None))[0]
n_supports01 = np.array(pd.read_csv('q'+ str(i) + 'n_supports.csv',header=None))[0]
trainerrs01 = np.array(pd.read_csv('q'+ str(i) + 'trainerrs.csv',header=None))[0]
ws01 = np.array(pd.read_csv('q'+ str(i) + 'ws.csv',header=None))[0]
# k = np.arange(2,N_relu,20)
k = np.arange(2,N_relu,2)
i=9

testerrs00 = np.array(pd.read_csv('q'+ str(i) + 'testerrs.csv',header=None))[0]
n_supports00 = np.array(pd.read_csv('q'+ str(i) + 'n_supports.csv',header=None))[0]
trainerrs00 = np.array(pd.read_csv('q'+ str(i) + 'trainerrs.csv',header=None))[0]
ws00 = np.array(pd.read_csv('q'+ str(i) + 'ws.csv',header=None))[0]
# k = np.arange(2,N_relu,20)
k = np.arange(2,N_relu,2)

#mediumseagreen
#cornflowerblue
#indianred
#orange


plt.figure(1)

plt.plot(k,testerrs00 , label='noise = 0',  linewidth=2,c='indianred')
plt.plot(k,testerrs01 , label='noise = 0.1',linewidth=2,c='cornflowerblue')
plt.plot(k,testerrs02 , label='noise = 0.2',linewidth=2,c='mediumseagreen')

plt.title("Test 01Loss with different noise")
plt.ylabel("Testerrors")
plt.xlabel("# Random ReLU Features")
# plt.axvline(thre,ls='-.', label='Interpolation Threshold',c='#FA7F6F')
plt.xlim(0,N_relu)
plt.legend()
plt.grid()
# plt.legend(title='Noise = 0.2')
plt.savefig('Test01Loss_SVM_relu_all.jpg', dpi=300)


plt.figure(2)
plt.plot(k,trainerrs00 , label='noise = 0',linewidth=2,c='indianred')
plt.plot(k,trainerrs01 , label='noise = 0.1',linewidth=2,c='cornflowerblue')
plt.plot(k,trainerrs02 , label='noise = 0.2',linewidth=2,c='mediumseagreen')
plt.title("Train 01Loss with different noise")
plt.ylabel("Trainerrors")
plt.xlabel("# Random ReLU Features")
# plt.axvline(thre,ls='-.', label='Interpolation Threshold',c='#FA7F6F')
plt.xlim(0,N_relu)
plt.legend()
plt.grid()
# plt.legend(title='Noise = 0.2')
plt.savefig('Train01Loss_SVM_relu_all.jpg', dpi=300)


plt.figure(3)
plt.plot(k,n_supports00 , label='noise = 0',linewidth=2,c='indianred')
plt.plot(k,n_supports01 , label='noise = 0.1',linewidth=2,c='cornflowerblue')
plt.plot(k,n_supports02 , label='noise = 0.2',linewidth=2,c='mediumseagreen')

plt.title("The number of support vectors")
plt.ylabel("Support vectors")
plt.xlabel("# Random ReLU Features")
plt.xlim(0,N_relu)
plt.legend()
plt.grid()
# plt.legend(title='Noise = 0.2')
plt.savefig('supportvectors_SVM_relu_all.jpg', dpi=300)



plt.figure(4)
plt.plot(k,ws00 , label='noise = 0',linewidth=2,c='indianred')
plt.plot(k,ws01 , label='noise = 0.1',linewidth=2,c='cornflowerblue')
plt.plot(k,ws02 , label='noise = 0.2',linewidth=2,c='mediumseagreen')
plt.title("Norm of ws")
plt.ylabel("ws")
plt.xlabel("# Random ReLU Features")
# plt.axvline(thre,ls='-.', label='Interpolation Threshold',c='#FA7F6F')
plt.xlim(0,N_relu)
plt.legend()
plt.grid()
# plt.legend(title='Noise = 0.2')
plt.savefig('Norm_SVM_relu_all.jpg', dpi=300)
plt.show()
