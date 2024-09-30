# we study the boundary condition effect on the spectrum of the system
import numpy as np
import multiprocessing 
import matplotlib.pyplot as plt
from NHmodel import NHGAA_ME
import mpmath as mp
import pandas as pd

mp.dps=300
pi=mp.pi

bd_all=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

def SpectrumWithBoundaryCondition(bd):
    L=89
    v=1
    theta=pi/3
    H=NHGAA_ME(L,v,theta,bd)
    E,ER=mp.eig(H)
    return E

def multiprocessing_spectrum(bd_all):
    with multiprocessing.Pool(processes=5) as pool:
        E_all = pool.map(SpectrumWithBoundaryCondition, bd_all)
    return E_all

def PcTest():
    for i in range(len(bd_all)):
        bd=bd_all[i]
        E_all=SpectrumWithBoundaryCondition(bd)
        ReE=list()
        ImE=list()
        for m in E_all:
            ReE.append(mp.re(m))
            ImE.append(mp.im(m))
        ReE=np.array(ReE)
        ImE=np.array(ImE)
        plt.plot(ReE,ImE,'.',markersize=3)
    plt.show()



if __name__=='__main__':
    Res=multiprocessing_spectrum(bd_all)
    Data=pd.DataFrame(Res)
    Data.to_csv('EigenvalueData_BoundaryCondition.csv',index=False)
    #PcTest(Res)


    
    
