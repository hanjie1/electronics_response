import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

runno=15910
tp_init=0.5

def ResFunc(x, par0, par1, par2, par3):

    xx = x-par2

    A1 = 4.31054*par0
    A2 = 2.6202*par0
    A3 = 0.464924*par0
    A4 = 0.762456*par0
    A5 = 0.327684*par0

    E1 = np.exp(-2.94809*xx/par1)
    E2 = np.exp(-2.82833*xx/par1)
    E3 = np.exp(-2.40318*xx/par1)

    lambda1 = 1.19361*xx/par1
    lambda2 = 2.38722*xx/par1
    lambda3 = 2.5928*xx/par1
    lambda4 = 5.18561*xx/par1

    return par3+(A1*E1-A2*E2*(np.cos(lambda1)+np.cos(lambda1)*np.cos(lambda2)+np.sin(lambda1)*np.sin(lambda2))+A3*E3*(np.cos(lambda3)+np.cos(lambda3)*np.cos(lambda4)+np.sin(lambda3)*np.sin(lambda4))+A4*E2*(np.sin(lambda1)-np.cos(lambda2)*np.sin(lambda1)+np.cos(lambda1)*np.sin(lambda2))-A5*E3*(np.sin(lambda3)-np.cos(lambda4)*np.sin(lambda3)+np.cos(lambda3)*np.sin(lambda4)))*np.heaviside(xx,1)

outfp = "../results/avg_pulse_{}.bin".format(runno)
with open(outfp, 'rb') as fn:
     raw=pickle.load(fn)

avg=raw[0]
avg_err=raw[1]

A0_list=[]
tp_list=[]
t0_list=[]
bl_list=[]
for ilink in range(10):
    for ich in range(256):
        apulse = avg[ilink][ich]
        apulse_err = avg_err[ilink][ich]

        pmax = np.amax(apulse)
        maxpos = np.argmax(apulse)
        nbf = 10
        naf = 10
        pbl = apulse[maxpos-nbf]
        a_xx = np.array(range(nbf+naf))*0.5
        popt, pcov = curve_fit(ResFunc, a_xx, apulse[maxpos-nbf:maxpos+naf],maxfev= 10000,p0=[80000,tp_init,(nbf-2)*0.5,1000])
        apulse_exp = ResFunc(a_xx,popt[0],popt[1],popt[2],popt[3])
        A0_list.append(popt[0])
        tp_list.append(popt[1])
        t0_list.append(popt[2])
        bl_list.append(popt[3])
        
        plt.scatter(a_xx, apulse[maxpos-nbf:maxpos+naf], c='r')
        #plt.errorbar(a_xx, apulse[maxpos-5:maxpos+20]-pbl,yerr=apulse_err[maxpos-5:maxpos+20],fmt='o')
        xx = np.linspace(0,nbf+naf,100)*0.5
        plt.plot(xx, ResFunc(xx,popt[0],popt[1],popt[2],popt[3]))
        plt.xlabel('us')
        plt.ylabel('ADC')
        plt.title('link%d chan%d'%(ilink,ich))
        plt.text(8,pmax-1500,'A0=%.2f'%popt[0],fontsize = 15)
        plt.text(8,pmax-2500,'tp=%.2f'%popt[1],fontsize = 15)
        plt.text(8,pmax-3500,'t0=%.2f'%popt[2],fontsize = 15)
        plt.text(8,pmax-4500,'bl=%.2f'%popt[3],fontsize = 15)
        plt.savefig("plots/run%d_link%d_ch%d"%(runno,ilink,ich))
        plt.close()

outfp1 = "results/fit_{}.bin".format(runno)
with open(outfp1, 'wb') as fn:
     pickle.dump([A0_list, tp_list, t0_list, bl_list], fn)
