# quantum dynamics of mosaic critical region


import mpmath as mp
from NHmodel import NHGAA,NHGAA_ME
import numpy as np
from matplotlib import pyplot as plt
import random as rd
import matplotlib.gridspec as gridspec
import multiprocessing


mp.dps=200

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
    return Data, XmeanData, psit2

def MainEvolutionME(theta):
# the main function to study the time evolution 
    
    # Hamiltonian and parameters 
    L=144

    v=1
    bd=0
    
    theta=theta*mp.pi
    #   求解本征态与本征值    
    H=NHGAA_ME(L,v,theta,bd)
    psi=initial_state(L)

    # time parameters and time evlution
    T=2000
    dt=16
    Data,XmeanData,psit2=TimeEvolutionRenormalization(H,psi,T,dt)

    Data1=np.array(Data,dtype=float)


    # plot the figure
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    # 第一个子图占据第一列
    ax1 = fig.add_subplot(gs[:, 0])  # : 表示从上到下全部行

    # 第二和第三个子图在第二列
    ax2 = fig.add_subplot(gs[0, 1])  # 第一行的第二列
    ax3 = fig.add_subplot(gs[1, 1])  # 第二行的第二列


    im1=ax1.imshow(Data1, cmap='hot')
    ax1.set_xlabel('L')
    ax1.set_ylabel(f't/{dt}')
    ax1.set_title(f'TimeEvolution_L_{L}_v_{v}_theta_{theta}_bd_{bd}')
    fig.colorbar(im1, ax=ax1) 

    ax2.plot(XmeanData)
    ax2.set_xlabel(f't/{dt}')
    ax2.set_ylabel('<x^2>')

    ax3.plot(psit2)
    ax3.set_xlabel('L')
    ax3.set_ylabel('psi^2')
    ax3.set_title('long-time distribution')

    filename=f'TimeEvolution_L_{L}_v_{v}_theta_{theta}_bd_{bd}_T_{T}.png'
    plt.savefig(filename)
    return 


def MainMultiprocessing():
# multiprocessing to calculte the time scale
    with multiprocessing.Pool(processes=5) as pool:
        # 使用map函数将任务分配给进程池
        theta_all=[0,1/3,1/2,4/3,1]
        results=pool.map(MainEvolutionME,theta_all)
    return



if __name__=='__main__':
    #MainEvolutionME()
    MainMultiprocessing()