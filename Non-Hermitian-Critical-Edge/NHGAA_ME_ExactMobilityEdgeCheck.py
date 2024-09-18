# 我们给出Exact Mobility edge的数据结果对比

import mpmath as mp
import numpy as np
from matplotlib import  pyplot as plt



mp.dps=200
pi=mp.pi

def ExactGamma(Lambda,theta):

    X=[mp.mpf(-3+i/100)  for i in range(600)]
    
    Y=[mp.mpc(-3*1j+i*1j/100)  for i in range(600)]

    Y1=[mp.mpf(-3+i/100)  for i in range(600)]
    
    Gamma=mp.zeros(600)

    for m in range(len(X)):
        for n in range(len(Y)):
            E=X[m]+Y[n]
            f=max(abs(2*E+mp.sqrt(4*E*E-4*mp.exp(2*1j*theta)*Lambda)),abs(2*E-mp.sqrt(4*E*E-4*mp.exp(2*1j*theta)*Lambda)))
            Gamma[m,n]=mp.re(mp.log(f)-mp.log(2)+mp.log(Lambda))
    return X,Y1,Gamma



if __name__=='__main__':
    Lambda=1
    theta=pi/3
    X,Y,Gamma=ExactGamma(Lambda,theta)
    X=np.array(X,dtype='float')
    Gamma=np.array(Gamma,dtype='float')
    Y=np.array(Y,dtype='float')

    Data=np.zeros([600,600])
    for m in range(600):
        for n in range(600):
            Data[m,n]=Gamma[m+600*n]
    

    contour_levels = [0, 0.5,1,1.5]
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Data,levels=contour_levels)
    ax.clabel(CS, inline=True, fontsize=10)
    plt.show()
        
    


