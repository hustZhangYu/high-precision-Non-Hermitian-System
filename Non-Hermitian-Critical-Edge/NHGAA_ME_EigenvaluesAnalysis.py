# We calculate and analyze the eigenstates and eigenvalues  of the Non Hermitian physics

import mpmath as mp
from NHmodel import NHGAA, NHGAA_ME
from matplotlib import pyplot as plt
import numpy as np
from NHPlot import PlotIPR
import multiprocessing 


mp.dps=300
pi=mp.pi


def GetEigenvalue(H):
    # this function can be used to calculate the eigenvalue of the spectrum
    # we returnt the eigenvalue and a figure which contains all the eigenvalues in the complex plane
    E,ER=mp.eig(H)
    
    X=list()
    Y=list()
    for m in E:
        X.append(mp.re(m))
        Y.append(mp.im(m))

    print(E)
    print(X)
    print(Y)
    plt.plot(X,Y,'.')
    plt.show()
    return E,ER


def Main():
    #主程序
#  initial parameters
    L=610
    v_all=[0.1,0.3,0.5,0.7,1.5]
    theta_all=[0,pi/3,pi/2,pi]
    bd=0
#  循环作图
    for v  in v_all:
        for theta in theta_all:

        #   求解本征态与本征值    
            H=NHGAA_ME(L,v,theta,bd)
            E,E1=mp.eig(H)

        #   画图与保存
            plt.figure()
            sc=PlotIPR(E,E1)
            plt.title(f'parameters:L={L},v={v},theta={theta},bd={bd}')
            plt.clim([0,1])
            plt.xlabel('Re(E)')
            plt.ylabel('Im(E)')
            plt.colorbar(sc)
            filename=f'fig_L_{L}_v_{v}_theta_{theta}_bd_{bd}.png'
            plt.savefig(filename)
            plt.close()

    return


def func(v):
# 该函数与并行的代码共同使用
# v 输入的参数代表pool当中用于并行计算的参数
    L=10
    theta_all=[0,pi/3,pi/2,pi]
    bd=0
    for theta in theta_all:
    #   求解本征态与本征值    
        H=NHGAA_ME(L,v,theta,bd)
        E,E1=mp.eig(H)

    #   画图与保存
        plt.figure()
        sc=PlotIPR(E,E1)
        plt.title(f'parameters:L={L},v={v},theta={theta},bd={bd}')
        plt.clim([0,1])
        plt.xlabel('Re(E)')
        plt.ylabel('Im(E)')
        plt.colorbar(sc)
        filename=f'fig_L_{L}_v_{v}_theta_{theta}_bd_{bd}.png'
        plt.savefig(filename)
        plt.close()
    return
    
def MultiprocessingCalcualtion():
#  为解决个人电脑运行慢的问题，服务器进行并行计算
#  采用了python当中的multiprocessing 中的pool
    with multiprocessing.Pool(processes=5) as pool:
        # 使用map函数将任务分配给进程池
        v_all=[0.1,0.3,0.5,0.7,1.5]
        results=pool.map(func,v_all)
    return


if __name__=='__main__':

    MultiprocessingCalcualtion()
