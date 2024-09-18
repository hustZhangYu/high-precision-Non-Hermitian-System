# 非厄米系统的量子动力学演化
# 根据非厄米物理的特性，我们全程使用mpmth做高精度计算
# mpmath的缺点在于，没有很好的定义矩阵运算


import mpmath as mp
from NHmodel import NHGAA,NHGAA_ME
import numpy as np
from matplotlib import pyplot as plt
import random as rd


mp.dps=200

def EscapeProbability(H,psi,T):
# In this function, we calculte the escape probability at different time  
# we need to calculate the integral 
# parameters : H: non-Hermitian Hamiltonian; dt: time interval; psi:initial state; T: whole time 
# 注意H的数据类型需要是mpmath库中的mpc或者mpf  
# # 对于积分方程来说，出去可以直接有限差分，我们还可以通过计算格林函数得到（换到频率空间和k空间）  
    dt=1
    U=mp.expm(-1j*H*dt)
    t=0
    while t<T:
        psi=U@psi
        t=t+dt
    return psi


def initial_state(L):
#  construct an initial state with given system size
    psi=mp.zeros(L,1)
    psi[mp.floor(L/2),0]=1
    return psi


def TimeEvolutionRenormalization(H,psi,T,dt):
#   Calculate the time evolution of the initial state under the Non-Hermitian Physics
#   we renormalize the wavefunction after a time interval
#   parameters: H:Hamiltonian, psi: initial state, T: all time, dt: time interval 
    
    # prepartion 
    L=len(psi)
    psit=psi
    Data=list()
    XmeanData=list()
    psit2=list()
    Xmean=0
    count=-L/2
    for m in psit:
        psit2.append(mp.re(m*mp.conj(m)))
        Xmean=Xmean+mp.re(m*mp.conj(m))*(count*count)
        count=count+1
    Data.append(psit2)
    XmeanData.append(Xmean)

    # unitary operator
    U=mp.expm(-1j*H*dt)
    t=0
    while t+dt<T:
        psit=U@psit
        s=0
        for m in psit:
            s=s+m*mp.conj(m)
        psit2=list()
        Xmean=0
        count=-L/2
        for m in psit:
            psit2.append(mp.re(m*mp.conj(m)/s))
            Xmean=Xmean+mp.re(m*mp.conj(m)/s)*(count*count)
            count=count+1
        Data.append(psit2)
        XmeanData.append(Xmean)
        t=t+dt
    return Data, XmeanData

def MainEvolutionME():
# the main function to study the time evolution 
    
    # Hamiltonian and parameters 
    L=144

    v=1
    bd=0
    theta=0*mp.pi/2
    #   求解本征态与本征值    
    H=NHGAA_ME(L,v,theta,bd)
    psi=initial_state(L)

    # time parameters and time evlution
    T=200
    dt=1
    Data,XmeanData=TimeEvolutionRenormalization(H,psi,T,dt)

    Data1=np.array(Data,dtype=float)
    # plot the figure
    plt.subplot(1,2,1)
    plt.imshow(Data1)
    plt.subplot(1,2,2)
    plt.plot(XmeanData)
    plt.show()
    return 


def MainEvolution():

    # Hamiltonian and parameters 
    L=144
    mu=1.5
    V=0
    phi=0
    alpha=0.5
    bd=1
    H=NHGAA(L,mu,V,phi,alpha,bd)
    psi=initial_state(L)

    # time parameters and time evlution
    T=200
    dt=1
    Data,XmeanData=TimeEvolutionRenormalization(H,psi,T,dt)

    Data1=np.array(Data,dtype=float)
    # plot the figure
    plt.subplot(1,2,1)
    plt.imshow(Data1)
    plt.subplot(1,2,2)
    plt.plot(XmeanData)
    plt.show()
    return 



def MainEscapeProbability():
# the main function to study the escape probability

    L=100
    mu=0.5
    V=0
    phi=0
    alpha=0
    bd=1
    H=NHGAA(L,mu,V,phi,alpha,bd)
    psi=initial_state(L)
    psi_t=EscapeProbability(H,psi,10)
    psi1=list()
    for m in psi_t:
        psi1.append(mp.re(m*mp.conj(m)))
    psi1=np.array(psi1)
    plt.plot(psi1)
    plt.show()
    return 

if __name__=='__main__':
    MainEvolutionME()