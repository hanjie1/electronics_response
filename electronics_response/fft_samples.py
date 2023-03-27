import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

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

dt = 0.5e-6 # MHz

for ilink in range(1):
    for ich in range(1):
        apulse = avg[ilink][ich]
        pmax = np.amax(apulse)
        maxpos = np.argmax(apulse)
        nn = len(apulse)

        nbf = 50
        naf = 50 
        tmp_pl = apulse[maxpos-nbf:maxpos+naf]
        new_nn = len(tmp_pl)

        a_xx = np.array(range(new_nn))*0.5e-6

        popt, pcov = curve_fit(ResFunc, a_xx, tmp_pl,maxfev= 10000,p0=[80000,2e-6,2e-5,1000])

        a_xx_1 = np.linspace(0,new_nn, 500)*0.5e-6
        apulse_fit = ResFunc(a_xx_1,popt[0],popt[1],popt[2],popt[3])

        fig,axes = plt.subplots(2,2,figsize=(12,6))
        axes[0,0].plot(a_xx,tmp_pl,marker='.')
        axes[0,0].plot(a_xx_1, apulse_fit, c='r')
        axes[0,0].text(0.6,0.9,'A0=%.2E'%popt[0],fontsize = 12, transform=axes[0,0].transAxes)
        axes[0,0].text(0.6,0.8,'tp=%.2E'%popt[1],fontsize = 12, transform=axes[0,0].transAxes)
        axes[0,0].text(0.6,0.7,'t0=%.2E'%popt[2],fontsize = 12, transform=axes[0,0].transAxes)
        axes[0,0].text(0.6,0.6,'Ab=%.2E'%popt[3],fontsize = 12, transform=axes[0,0].transAxes)

        a_fft = np.abs(np.fft.fft(tmp_pl))
        freq = np.fft.fftfreq(len(tmp_pl), d=dt)
        #axes[0,1].plot(freq[1:new_nn//2],a_fft[1:new_nn//2],marker='.')
        axes[0,1].plot(freq,a_fft,marker='.')

        new_freq = freq
#        print(freq)
        new_fft = a_fft 
        extrapos=0
        for i in range(len(freq)//2-1):
            tmp_f = freq[i]+(freq[i+1]-freq[i])/2
            tmp_a = a_fft[i]+(a_fft[i+1]-a_fft[i])/2

            new_freq = np.insert(new_freq, 1+i+extrapos, tmp_f) 
            new_fft = np.insert(new_fft, 1+i+extrapos, tmp_a) 
            extrapos = extrapos+1

        for i in range(len(freq)//2,len(freq)-1):
            tmp_f = freq[i]+(freq[i+1]-freq[i])/2
            new_freq = np.insert(new_freq, 1+i+extrapos, tmp_f) 
            tmp_a = a_fft[i]+(a_fft[i+1]-a_fft[i])/2
            new_fft = np.insert(new_fft, 1+i+extrapos, tmp_a) 
            extrapos = extrapos+1

        axes[1,0].plot(new_freq,new_fft)

#        print(new_freq)
        new_pl = np.abs(np.fft.irfft(new_fft))

        axes[1,1].plot(range(len(new_pl)),new_pl)
plt.show()
