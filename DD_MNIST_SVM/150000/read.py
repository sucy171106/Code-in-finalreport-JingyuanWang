import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
N_relu = 1500

i = 0
testerrs = np.array(pd.read_csv('q'+ str(i) + 'testerrs.csv',header=None))[0]

n_supports = np.array(pd.read_csv('q'+ str(i) + 'n_supports.csv',header=None))[0]
trainerrs = np.array(pd.read_csv('q'+ str(i) + 'trainerrs.csv',header=None))[0]
ws = np.array(pd.read_csv('q'+ str(i) + 'ws.csv',header=None))[0]
k = np.arange(2,N_relu,1)
thre = 115
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
plt.savefig('TestTrain01Loss_SVM_relu00.jpg', dpi=300)




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
plt.savefig('supportvectors_SVM_relu00.jpg', dpi=300)


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
plt.savefig('Norm_SVM_relu00.jpg', dpi=300)
plt.show()

