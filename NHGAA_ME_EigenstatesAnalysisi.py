# We calculate and analyze the eigenstates and eigenvalues  of the Non Hermitian physics

import mpmath as mp
from NHmodel import NHGAA, NHGAA_ME
from matplotlib import pyplot as plt
import numpy as np
from NHPlot import PlotIPR
import multiprocessing 
import pandas as pd

mp.dps=400
pi=mp.pi
L_all=[144,233,377,610,987]



def GetIPR(psi):
    IPR=0
    for n in psi:
        n1=mp.re(n*mp.conj(n))
        IPR=IPR+n1*n1
    return IPR

def GEtIPRMosaic(psi):
    IPR=0
    n=0
    while n+1<=len(psi)-1:
        n1=mp.re(psi[n]*mp.conj(psi[n]))+mp.re(psi[n+1]*mp.conj(psi[n+1]))
        IPR=IPR+n1*n1
        n=n+2
    return IPR

def GetEigenstate(v):
#   with this function, we plot the eigenstate with the maxium imaginary part
    L=610
    bd=1
    theta_all=[0,pi/3,pi/2,pi]

    for theta in theta_all:
        H=NHGAA_ME(L,v,theta,bd)
        E,EL,ER=mp.eig(H,left=True,right=True)
        for m in range(L):
            E[m]=-1j*E[m]
        # eig_sort 排序默认是对实部，因此我们给前面乘以一个虚数因子，变成按照虚部大小排列
        E,EL,ER=mp.eig_sort(E,EL,ER)


        s=[0,100,300,500,609]
        #s=[0,1,3,5,9]
        plt.figure()
        for n in range(len(s)):
            psi=ER[:,s[n]]
            psi1=list()
            Norm=0
            for m in psi:
                psi1.append(mp.re(m*mp.conj(m)))
                Norm=Norm+mp.re(m*mp.conj(m))
            psi1=np.array(psi1)
            psi1=psi1/Norm
            plt.subplot(len(s),1,n+1)
            plt.plot(psi1)
            plt.title(f'parameters:L={L},v={v},theta={theta},bd={bd},n={s[n]}')
            plt.xlabel('L')
            plt.ylabel('psi^2')
        filename=f'TypicalEigenstates_L_{L}_v_{v}_theta_{theta}_bd_{bd}.png'
        plt.subplots_adjust(hspace=2.5)
        plt.savefig(filename)
        plt.close()

    return 
  
def MultiprocessingCalcualtion1():
#  为解决个人电脑运行慢的问题，服务器进行并行计算
#  采用了python当中的multiprocessing 中的pool
    with multiprocessing.Pool(processes=5) as pool:
        # 使用map函数将任务分配给进程池
        v_all=[0.1,0.3,0.5,0.7,1.5]
        results=pool.map(GetEigenstate,v_all)
    return

def EigenstatesScaling(n):
# 我们计算不同尺寸下本征态的IPR，我们首先将本征态按照IPR值的大小进行排序，然后考虑不同的尺寸
    bd=1
    #theta_all=[0,pi/3,pi/2,pi]
    theta=2*pi/(n+1)
    v=1

    for L in L_all:
        H=NHGAA_ME(L,v,theta,bd)
        E,ER=mp.eig(H)

        IPRData=list()
        for n in range(L):
            psi=ER[:,n]
            psi1=list()
            Norm=0
            for m in psi:
                psi1.append(m)
                Norm=Norm+mp.re(m*mp.conj(m))
            for m in range(len(psi1)):
                psi1[m]=psi1[m]/mp.sqrt(Norm)
            psi1=np.array(psi1)
            IPRData.append(-mp.log(GEtIPRMosaic(psi1))/mp.log(mp.floor(L/2)))    
        IPRData.sort()
        X=[i/L for i in range(L)]
        plt.plot(X,IPRData)
    plt.xlabel('n/L')
    plt.ylabel('IPR')
    filename=f'IPRScaling_v_{v}_theta_{theta}.png'
    plt.savefig(filename)
    #plt.show()
    
    return

def MultiProcessingEigenstateScaling():
    with multiprocessing.Pool(processes=5) as pool:
        # 使用map函数将任务分配给进程池
        v_all=[0,1,2,3,4]
        results=pool.map(EigenstatesScaling,v_all)
    return

def SingleEigenstateScaling(L):
    # we choose a single state and calculate its IPR in different size
    # we choose  eigenstates with different system sizes but the same position after we sort the IPR
    bd=1
    v=1
    theta=2*pi/3
    H=NHGAA_ME(L,v,theta,bd)
    E,ER=mp.eig(H)
    IPRData=list()
    for n in range(L):
        psi=ER[:,n]
        psi1=list()
        Norm=0
        for m in psi:
            psi1.append(m)
            Norm=Norm+mp.re(m*mp.conj(m))
        for m in range(len(psi1)):
            psi1[m]=psi1[m]/mp.sqrt(Norm)
        psi1=np.array(psi1)
        IPRData.append(-mp.log(GEtIPRMosaic(psi1))/mp.log(mp.floor(L/2)))

    res=list()
    for x in range(int(mp.floor(L/8)),int(mp.floor(3*L/8))):
        res.append(IPRData[x])
    return res

def MultiProcessingSingleEigenstateScaling():
    with multiprocessing.Pool(processes=5) as pool:
        # 使用map函数将任务分配给进程池
        L_all=[144,233,377,610,987]
        # L_all=[20,30,40,60,80]
        results=pool.map(SingleEigenstateScaling,L_all)
    res1=pd.DataFrame(results)
    res1.to_csv('SingleEigenstateScalingMulti.csv')
    return

def ShowEigenstates(theta):
    L = 144
    bd = 1
    v = 1
    H = NHGAA_ME(L, v, theta, bd)
    E, ER = mp.eig(H)
    for n in range(L):
        psi = ER[:, n]
        res = []
        for m in psi:
            # 将 mpmath 的复数转换为 Python 的浮点数
            res.append(float(mp.re(m * mp.conj(m))))
        plt.fill_between(range(len(res)), res, color='blue', alpha=0.1)
    plt.xlabel('L')
    plt.ylabel(f'$|\psi|^2$')
    plt.title(f'L={L},v={v},theta={theta},bd={bd}')
    plt.ylim(0,1)
    plt.savefig(f'AllEigenstates_L_{L}_v_{v}_theta_{theta}_bd_{bd}.png')
    plt.close()
    return
        
def MultiProcessingShowEigenstates():
    with multiprocessing.Pool(processes=5) as pool:
        # 使用map函数将任务分配给进程池
        theta_all=[-pi/3 ,0,pi/3,pi/2,pi]
        results=pool.map(ShowEigenstates,theta_all)
    return

if __name__=='__main__':
    MultiProcessingShowEigenstates()
    # ShowEigenstates(pi/3)
    # SingleEigenstateScaling(610)
    # MultiProcessingSingleEigenstateScaling()
    # MultiProcessingEigenstateScaling()
    # EigenstatesScaling()
    # MultiprocessingCalcualtion()
    # GetEigenstate(1)
    
    
