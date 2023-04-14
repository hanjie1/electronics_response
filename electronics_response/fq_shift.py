import numpy as np
import matplotlib.pyplot as plt
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

xx=np.linspace(0,10,30)
par0=80000
par1=0.5
par2=0.2
par3=1000
yy = ResFunc(xx,par0,par1,par2,par3)

fft_0 = np.fft.rfft(yy)
freq_0 = np.fft.rfftfreq(len(xx), d=1)

par2_1=3
#xx_1=np.linspace(0,10,25)
#yy_1 = ResFunc(xx,par0,par1,par2,par3)
#yy_1 = np.concatenate(([yy_1[0]]*par2_1,yy_1[:-par2_1]))
#xx_1 = xx+par2_1
yy_1 = np.concatenate(([yy[0]]*par2_1,yy[:-par2_1]))
#yy_1 = yy

fft_1 = np.fft.rfft(yy_1)
freq_1 = np.fft.rfftfreq(len(yy_1), d=1)

rr_list=[]
im_list=[]
im_1 = []
im_2 = []

for ii in range(1,len(freq_0)):
    ff = freq_0[ii]
    ff_ratio = fft_1[ii]/fft_0[ii]
    rr_ratio = cmath.polar(ff_ratio)[0]
    im_diff = cmath.polar(ff_ratio)[1]

    rr_list.append(rr_ratio)

    im_pred = 2j*np.pi*ff*(par2_1 - par2)
    im_pred_polar = cmath.polar(np.exp(im_pred))[1]
    im_ratio = im_diff/im_pred_polar
    if abs(im_pred_polar)<1e-3:
       continue
    im_list.append(im_ratio)
    im_1.append(im_diff)
    im_2.append(im_pred_polar)

new_fft = [fft_1[i]*np.exp(2j*np.pi*freq_1[i]*(par2_1 - par2)) for i in range(len(fft_1))]
new_pl = np.fft.irfft(new_fft)

fig,axes = plt.subplots(2,3,figsize=(12,10))
axes[0,0].plot(xx,yy,marker='.')
axes[0,0].plot(xx,yy_1,marker='.')
axes[0,0].plot(xx,new_pl,marker='.')
axes[0,1].plot(range(len(rr_list)),rr_list,marker='.')
axes[0,2].plot(range(len(im_list)),im_list,marker='.')
axes[1,0].plot(range(len(im_1)),im_1,marker='.')
axes[1,1].plot(range(len(im_2)),im_2,marker='.')
plt.show()

