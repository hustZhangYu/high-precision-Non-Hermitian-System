# In this code, we construct the Hamiltonain with given parameters.
# 注意： 我们使用了mpmath进行可调精度的计算 mp.dps=100 表示位数
# mpmath 不存在矩阵运算结构，所以写循环略显繁琐

import numpy as np
from scipy.linalg import expm
import random as rd
import math
from matplotlib import pyplot as plt
import mpmath as mp
mp.dps=200

pi=mp.pi
def NHGAA(L,mu,V,phi,alpha,bd):
    # alpah : hopping strength 
    # bd: boundary condition 
    omega=(mp.sqrt(5)-1)/2
    H=mp.zeros(L,L)
    t=mp.zeros(1,L-1)
    for m in range(L-1):
        t[m]=1+mu*mp.cos(2*pi*omega*m+pi*omega+phi)
    for i in range(L-1):
        H[i,i+1]=t[i]*math.exp(alpha)
        H[i+1,i]=t[i]*math.exp(-alpha)
    
    H[L-1,0]=bd*math.exp(alpha)
    H[0,L-1]=bd*math.exp(-alpha)

    for i in range(L):
        H[i,i]=V*mp.cos(2*pi*omega*(i+1)+phi)

    return H

def NHGAA_ME(L,v,alpha=0,bd=0):
    # Here we give the Hamiltonian with exact critical-localized edges
    # The properties of the Hamiltonian :see paper PRL 131,176401
    # constant
    pi=mp.pi
    omega=(mp.sqrt(5)-1)/2
    phi=rd.random()*2*pi
    phi=0


    # hopping and onsite terms
    H=mp.zeros(L,L)
    t=mp.zeros(1,L)
    V=mp.zeros(1,L)
    for m in range(L):
        if m%2==0:
            V[m]=mp.cos(2*pi*omega*m+phi)
            t[m]=mp.cos(2*pi*omega*m+phi)
        if m%2==1:
            V[m]=mp.cos(2*pi*omega*(m-1)+phi)
            t[m]=mp.exp(1j*alpha)*v
    
    # construct the Hamiltonian
    for i in range(L-1):
        H[i,i+1]=t[i]
        H[i+1,i]=t[i]
        
    H[L-1,0]=bd*t[i]
    H[0,L-1]=bd*t[i]

    for i in range(L):
        H[i,i]=V[i]

    return H


def eigenstate_entropy(psi):
    # 计算本征态纠缠熵，类似于描述局域扩展的物理量性质
    res=0
    for m in psi:
        n=mp.re(m*mp.conj(m))
        res=res-n*mp.log(n)
    return mp.re(res) 





if __name__=='__main__':

    L=1
    mu=0.5
    V=0
    phi=0
    alpha=0.5
    bd=1
    H=NHGAA(L,mu,V,phi,alpha,bd)
    E,E1=mp.eig(H)

    psi=E1[:,1]

    m=eigenstate_entropy(psi)
    print(m)

    a=list()
    b=list()
    for m in range(len(E)):
        a.append(mp.re(E[m]))
        b.append(mp.im(E[m]))
    plt.subplot(1,2,1)
    plt.plot(a,b,'.')
    plt.ylim(-1,1)

    for m in range(L):
        psi=E1[:,m]
        a=list()
        a1=0
        for n in psi:
            a.append(mp.re(mp.conj(n)*n))
            a1=a1+mp.re(mp.conj(n)*n)
        for n in range(len(a)):
            a[n]=a[n]/a1
        plt.subplot(1,2,2)
        plt.plot(a)
    plt.show()
    
        
    










