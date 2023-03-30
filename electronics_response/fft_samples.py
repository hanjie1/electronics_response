import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import cmath

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

runno=15155

outfp = "results/avg_pulse_{}.bin".format(runno)
with open(outfp, 'rb') as fn:
     raw=pickle.load(fn)

avg=raw[0]
avg_err=raw[1]

dt = 0.5 # us
tp_init = 2 #us

for ilink in range(1):
    for ich in range(1):
        apulse = avg[ilink][ich]
        pmax = np.amax(apulse)
        maxpos = np.argmax(apulse)
        nn = len(apulse)

        nbf = 10
        naf = 8 
        tmp_pl = apulse[maxpos-nbf:maxpos+naf]
        new_nn = len(tmp_pl)

        a_xx = np.array(range(new_nn))*dt

        popt, pcov = curve_fit(ResFunc, a_xx, tmp_pl,maxfev= 10000,p0=[80000,tp_init,(nbf-8)*0.5,1000])

        a_xx_1 = np.linspace(0,new_nn*0.5, 500)
        apulse_fit = ResFunc(a_xx_1,popt[0],popt[1],popt[2],popt[3])

        fig,axes = plt.subplots(1,2,figsize=(12,6))
        axes[0].plot(a_xx,tmp_pl,marker='.')
        axes[0].plot(a_xx_1, apulse_fit, c='r')
        axes[0].text(0.7,0.9,'A0=%.2f'%popt[0],fontsize = 12, transform=axes[0].transAxes)
        axes[0].text(0.7,0.8,'tp=%.2f'%popt[1],fontsize = 12, transform=axes[0].transAxes)
        axes[0].text(0.7,0.7,'t0=%.2f'%popt[2],fontsize = 12, transform=axes[0].transAxes)
        axes[0].text(0.7,0.6,'Ab=%.2f'%popt[3],fontsize = 12, transform=axes[0].transAxes)
        axes[0].set_title("original pulse")

        a_fft = np.fft.rfft(apulse[maxpos-nbf:maxpos+20])
        new_nn_1 = nbf+20
        freq = np.fft.rfftfreq(new_nn_1, d=dt)

        new_dt = 0.5
        new_fft_1 = [a_fft[i]*np.exp(2j*np.pi*freq[i]*popt[2]) for i in range(len(a_fft))]
        nextra = int(new_nn_1/new_dt-new_nn_1)
        new_fft_2 = np.concatenate((new_fft_1,np.zeros(nextra//2)),dtype = "complex_")


        new_pl = np.fft.irfft(new_fft_2)
        new_pl = new_pl*tmp_pl[0]/new_pl[0]

        new_tt = np.arange(len(new_pl))*new_dt*dt
        axes[1].plot(new_tt,new_pl, marker='.')
        plt.show()

