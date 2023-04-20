import pickle
import numpy as np
import matplotlib.pyplot as plt

runno=15155

outfp = "results/fit_{}.bin".format(runno)
with open(outfp, 'rb') as fn:
     raw=pickle.load(fn)

A0_list=raw[0]
tp_list=raw[1]
t0_list=raw[2]
bl_list=raw[3]
chi2_list=raw[4]

xx = range(2560)
fig,axes = plt.subplots(2,2,figsize=(12,8))
axes[0,0].plot(xx, A0_list)
axes[0,1].plot(xx, bl_list)
axes[1,0].plot(xx, tp_list)
axes[1,1].plot(xx, t0_list)

fig1,ax1 = plt.subplots(figsize=(12,8))
ax1.plot(xx, chi2_list)
plt.show()
