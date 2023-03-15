import numpy as np
import matplotlib.pyplot as plt

runno=15155

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

xx=np.linspace(-5,20,100)
par0=80000
par1=2
par2=2
par3=0
yy = ResFunc(xx,par0,par1,par2,par3)
plt.plot(xx,yy,marker='.')
plt.show()

