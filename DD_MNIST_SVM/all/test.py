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

k = np.arange(2,N_relu,2)



#mediumseagreen
#cornflowerblue
#indianred
#orange

plt.figure(1)

plt.plot(k,testerrs00 , label='noise = 0',  linewidth=2,c='indianred')

plt.title("Test 01Loss with different noise")
plt.ylabel("Testerrors")
plt.xlabel("# Random ReLU Features")
# plt.axvline(thre,ls='-.', label='Interpolation Threshold',c='#FA7F6F')
plt.xlim(0,N_relu)
plt.legend()
plt.grid()
# plt.legend(title='Noise = 0.2')
plt.savefig('Test01Loss_SVM_relu_all.jpg', dpi=300)


