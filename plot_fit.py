import pickle
import numpy as np
import matplotlib.pyplot as plt

runno=15155

outfp = "results/avg_pulse_{}.bin".format(runno)
with open(outfp, 'rb') as fn:
     avg=pickle.load(fn)

outfp = "results/A0_{}.bin".format(runno)
with open(outfp, 'rb') as fn:
     A0=pickle.load(fn)

outfp = "results/tp_{}.bin".format(runno)
with open(outfp, 'rb') as fn:
     tp=pickle.load(fn)

avg=np.array(avg)
print(avg.shape)
print(A0.shape)
print(tp.shape)

plt.plot(range(2560), A0)
plt.show()


