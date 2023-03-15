import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

runno=15155

def ResFunc(x, par0, par1):
    A1 = 4.31054*par0
    A2 = 2.6202*par0
    A3 = 0.464924*par0
    A4 = 0.762456*par0
    A5 = 0.327684*par0

    E1 = np.exp(-2.94809*x/par1)
    E2 = np.exp(-2.82833*x/par1)
    E3 = np.exp(-2.40318*x/par1)

    lambda1 = 1.19361*x/par1
    lambda2 = 2.38722*x/par1
    lambda3 = 2.5928*x/par1
    lambda4 = 5.18561*x/par1

    return A1*E1-A2*E2*(np.cos(lambda1)+np.cos(lambda1)*np.cos(lambda2)+np.sin(lambda1)*np.sin(lambda2))+A3*E3*(np.cos(lambda3)+np.cos(lambda3)*np.cos(lambda4)+np.sin(lambda3)*np.sin(lambda4))+A4*E2*(np.sin(lambda1)-np.cos(lambda2)*np.sin(lambda1)+np.cos(lambda1)*np.sin(lambda2))-A5*E3*(np.sin(lambda3)-np.cos(lambda4)*np.sin(lambda3)+np.cos(lambda3)*np.sin(lambda4))

def chi_squared(y_mea, y_mea_std, y_exp):
# Calculate the chi-squared statistic
    chi2_stat = np.sum(((y_mea - y_exp)/y_mea_std)**2)
    
    # Calculate the degrees of freedom
    dof = len(y_mea) - 2
    chi2_stat = chi2_stat/dof
    
    # Calculate the p-value using the chi2 distribution
#    p_value = 1 - chi2.cdf(chi2_stat, dof)
#    print(chi2_stat, p_value)
    
    return chi2_stat

outfp = "results_1/avg_pulse_{}.bin".format(runno)
with open(outfp, 'rb') as fn:
     raw=pickle.load(fn)

avg=raw[0]
avg_err=raw[1]

#A0_new = A0.reshape(-1)
#tp_new = tp.reshape(-1)
#fig,axes = plt.subplots(2,1)
#axes[0].scatter(range(len(A0_new)),A0_new,marker='.')
#axes[0].set_xlabel('chan',fontsize=12)
#axes[0].set_ylabel('A0',fontsize=12)
#
#axes[1].scatter(range(len(tp_new)),tp_new,marker='.')
#axes[1].set_xlabel('chan',fontsize=12)
#axes[1].set_ylabel('tp',fontsize=12)
##plt.tight_layout()
#plt.show()

#apulse = avg[0][0]
#apulse_err = avg_err[0][0]
#pmax = np.amax(apulse)
#maxpos = np.argmax(apulse)
#nbf = 5
#naf = 20
#pbl = apulse[maxpos-nbf]
#a_xx = np.array(range(nbf+naf))*0.5
#popt, pcov = curve_fit(ResFunc, a_xx, apulse[maxpos-nbf:maxpos+naf]-pbl)
#apulse_exp = ResFunc(a_xx,popt[0],popt[1])
##chi2,pval = chi_squared(apulse[maxpos-nbf:maxpos+naf]-pbl, apulse_exp)
#chi2 = chi_squared(apulse[maxpos-nbf:maxpos+naf]-pbl, apulse_err[maxpos-nbf:maxpos+naf], apulse_exp)
#
#nbf1 = 4
#naf1 = 20
#pbl1 = apulse[maxpos-nbf1]
#a_xx1 = np.array(range(nbf1+naf1))*0.5
#popt1, pcov1 = curve_fit(ResFunc, a_xx1, apulse[maxpos-nbf1:maxpos+naf1]-pbl1,p0=popt)
#apulse_exp1 = ResFunc(a_xx1,popt1[0],popt1[1])
##chi2_1,pval_1 = chi_squared(apulse[maxpos-nbf1:maxpos+naf1]-pbl1, apulse_exp1)
#chi2_1 = chi_squared(apulse[maxpos-nbf1:maxpos+naf1]-pbl1, apulse_err[maxpos-nbf1:maxpos+naf1], apulse_exp1)
#
#fig,axes = plt.subplots(1,2)
#axes[0].scatter(a_xx, apulse[maxpos-nbf:maxpos+naf]-pbl,c='r')
#xx = np.linspace(0,nbf+naf,100)*0.5
#axes[0].plot(xx, ResFunc(xx,popt[0],popt[1]),label='fit1')
#axes[0].text((nbf+naf-8)*0.5,pmax-1500,'A0=%.2f'%popt[0],fontsize = 15)
#axes[0].text((nbf+naf-8)*0.5,pmax-2500,'tp=%.2f'%popt[1],fontsize = 15)
#axes[0].text((nbf+naf-8)*0.5,pmax-3500,'chi2=%.2f'%chi2,fontsize = 15)
#axes[0].set_xlabel('us')
#axes[0].set_ylabel('ADC')
#axes[0].legend()
#
#xx1 = np.linspace(0,nbf1+naf1,100)*0.5
#axes[1].scatter(a_xx1, apulse[maxpos-nbf1:maxpos+naf1]-pbl1,c='r')
#axes[1].plot(xx1, ResFunc(xx1,popt1[0],popt1[1]),label='fit2')
#axes[1].text((nbf1+naf1-8)*0.5,pmax-1500,'A0=%.2f'%popt1[0],fontsize = 15)
#axes[1].text((nbf1+naf1-8)*0.5,pmax-2500,'tp=%.2f'%popt1[1],fontsize = 15)
#axes[1].text((nbf1+naf1-8)*0.5,pmax-3500,'chi2=%.2f'%chi2_1,fontsize = 15)
#axes[1].set_xlabel('us')
#axes[1].set_ylabel('ADC')
#axes[1].legend()
#
#plt.show()
#
chi2_list = []
for ilink in range(10):
    for ich in range(256):
        apulse = avg[ilink][ich]
        apulse_err = avg_err[ilink][ich]
        pmax = np.amax(apulse)
        maxpos = np.argmax(apulse)
        nbf = 5
        naf = 20
        pbl = apulse[maxpos-nbf]
        a_xx = np.array(range(nbf+naf))*0.5
        popt, pcov = curve_fit(ResFunc, a_xx, apulse[maxpos-nbf:maxpos+naf]-pbl)
        apulse_exp = ResFunc(a_xx,popt[0],popt[1])
        #chi2,pval = chi_squared(apulse[maxpos-nbf:maxpos+naf]-pbl, apulse_exp)
        chi2 = chi_squared(apulse[maxpos-nbf:maxpos+naf]-pbl, apulse_err[maxpos-nbf:maxpos+naf], apulse_exp)
        chi2_list.append(chi2)
#        
#        plt.scatter(a_xx, apulse[maxpos-5:maxpos+20]-pbl, c='r')
#        plt.errorbar(a_xx, apulse[maxpos-5:maxpos+20]-pbl,yerr=apulse_err[maxpos-5:maxpos+20],fmt='o')
#        xx = np.linspace(0,25,100)*0.5
#        plt.plot(xx, ResFunc(xx,popt[0],popt[1]))
#        plt.xlabel('us')
#        plt.ylabel('ADC')
#        plt.title('link%d chan%d'%(ilink,ich))
#        plt.text(8,pmax-1500,'A0=%.2f'%popt[0],fontsize = 15)
#        plt.text(8,pmax-2500,'tp=%.2f'%popt[1],fontsize = 15)
#        plt.text(8,pmax-3500,'chi2/dof=%.2f'%chi2,fontsize = 15)
#        plt.savefig("plots/run%d_link%d_ch%d"%(runno,ilink,ich))
#        plt.close()

plt.plot(range(2560),chi2_list)
plt.show()
