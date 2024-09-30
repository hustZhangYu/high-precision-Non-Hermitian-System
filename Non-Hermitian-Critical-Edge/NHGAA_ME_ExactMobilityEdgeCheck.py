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
            a=mp.log(abs(E/(mp.exp(1j*theta)*Lambda)+mp.sqrt(E*E/(mp.exp(2*1j*theta)*Lambda**2)-1)))
            b=mp.log(abs(E/(mp.exp(1j*theta)*Lambda)-mp.sqrt(E*E/(mp.exp(2*1j*theta)*Lambda**2)-1)))
            f=max(a,b) # we choose the smaller one, because the smaller one determines the lowest decay rate
            f=max(f,0)
            Gamma[m,n]=mp.re(f)/2
    return X,Y1,Gamma

def MainExactGamma():
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
    
    MinValue= np.min(Data)
    MaxValue= np.max(Data)
    print(f"Data的最小值是: {MinValue}")
    
    level = np.linspace(MinValue, MaxValue, 10)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Data,levels=level)
    ax.clabel(CS, inline=True, fontsize=10)
    plt.colorbar(CS, ax=ax)
    plt.show()
        

def CalculateTheLE_TransferMatrix():
    # 我们通过转移矩阵的方法来计算LE
    def TransferMatrix(E,n,theta,Lambda):
        # 计算能量为E,第n个元胞的转移矩阵
        omega=(mp.sqrt(5)-1)/2
        M=mp.zeros(2,2)
        M[0,0]=(E*E-2*E*mp.cos(2*pi*omega*n))/(mp.exp(1j*theta)*Lambda*mp.cos(2*pi*omega*n))
        M[0,1]=(-E+mp.cos(2*pi*omega*n))/(mp.cos(2*pi*omega*n))
        M[1,0]=(E-mp.cos(2*pi*omega*n))/(mp.cos(2*pi*omega*n))
        M[1,1]=-mp.exp(1j*theta)*Lambda/(mp.cos(2*pi*omega*n))
        return M
    E_all=[0.5,0.8,0.9,1.2]
    for E in E_all:
        theta=0   
        Lambda=1
        M=TransferMatrix(E,0,theta,Lambda)
        L=3000
        Data1=list()
        Data2=list()
        Data3=list()
        for n in range(1,L):
            M=M@TransferMatrix(E,n,theta,Lambda)
            eigvalue,eigvector=mp.eig(M)
            Data1.append(mp.log(abs(eigvalue[0]))/n)
            Data2.append(mp.log(abs(eigvalue[1]))/n)
            Data3.append(mp.log(abs(M[0,0]+M[1,1]))/n)
        # plt.plot(Data1,label='LE1')
        # plt.plot(Data2,label='LE2')
        plt.plot(Data3,label='Trace')
        plt.legend()
    plt.show()
        

if __name__=='__main__':

    CalculateTheLE_TransferMatrix()


