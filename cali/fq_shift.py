import numpy as np
from scipy.optimize import curve_fit
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

nbin = 21
tt = 10
# original pulse
xx=np.linspace(0,tt,nbin)
print(xx)
delt = tt/nbin 
par0=10000
par1=3
par2=3.2
par3=1000
yy = ResFunc(xx,par0,par1,par2,par3)
fft_o = np.fft.fft(yy)
freq_o = np.fft.fftfreq(len(xx), d=0.5)
print(freq_o)

# t0=0 func
xx_m = np.linspace(0,tt,200)
yy_m = ResFunc(xx_m,par0,par1,0,par3)

# shift by e^iwt
fft_1 = [fft_o[i]*np.exp(2j*np.pi*freq_o[i]*(par2)) for i in range(len(fft_o))]
new_pl = np.fft.ifft(fft_1)
new_pl = new_pl.real
new_xx = np.array(range(len(new_pl)))*0.5

# check if e^iwt is correct
xx_0 = xx
yy_0 = ResFunc(xx,par0,par1,0,par3)
fft_0 = np.fft.fft(yy_0)

rr_list=[]
im_list=[]

for ii in range(1,len(freq_o)):
    ff = freq_o[ii]

    ff_ratio = fft_0[ii]/fft_o[ii]
    rr_ratio = cmath.polar(ff_ratio)[0]
    im_diff = cmath.polar(ff_ratio)[1]

    rr_list.append(rr_ratio)

    im_pred = 2j*np.pi*ff*(par2)
    im_pred_polar = cmath.polar(np.exp(im_pred))[1]
    im_ratio = im_diff/im_pred_polar
    if abs(im_pred_polar)<1e-3:
       continue
    im_list.append(im_ratio)
    #im_1.append(im_diff)
    #im_2.append(im_pred_polar)

# shift by using ideal fft

fig,axes = plt.subplots()
axes.plot(xx,yy,marker='.',label='orignal pulse')
axes.plot(xx_m,yy_m,label='function with t0=0')
axes.plot(new_xx,new_pl,marker='.',label='shifted pulse')
axes.legend(fontsize="20")

fig1,axes1 = plt.subplots(1,3)
axes1[0].plot(xx,yy,marker='.',label='t0=3.2')
axes1[0].plot(xx,yy_0,marker='.',label='t0=0')
axes1[0].set_title("pulses in time domain")
axes1[0].legend()

axes1[1].plot(range(len(rr_list)),rr_list,marker='.')
axes1[1].set_title("FFT real part ratio")

axes1[2].plot(range(len(im_list)),im_list,marker='.')
axes1[2].set_title("FFT img part diff/2*pi*f*t0")
plt.show()

