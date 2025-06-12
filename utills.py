import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.axes import Axes as ax

import numpy as np
import tensorflow as tf
import function2D as fun
#define and plot functions(in original fn line 52-90)
def create2Dfunction(x1Lim=(-2.0, 2.0), x2Lim=(-2.0, 2.0), N=200):
    X1 = np.linspace(x1Lim[0],x1Lim[1],N)
    X2 = np.linspace(x2Lim[0],x2Lim[1],N)
    X1, X2 = np.meshgrid(X1, X2)
    Y = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            Y[i,j] = fun.f( x=np.array([X1[i,j],X2[i,j]]))
    return X1, X2, Y

def plotFunction2D(x1Lim=(-2.0, 2.0), x2Lim=(-2.0, 2.0), N=200):
    X1 = np.linspace(x1Lim[0],x1Lim[1],N)
    X2 = np.linspace(x2Lim[0],x2Lim[1],N)
    X1, X2 = np.meshgrid(X1, X2)
    print(X1.shape)
    
    Y = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            Y[i,j] = fun.f( x=np.array([X1[i,j],X2[i,j]]) )
    
    ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
    # xx= np.c_[X1, X2]
    # print("ccc--",xx.shape)
    # Y=fun.f(X1)
    CS = ax.contour(X1, X2, Y)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
    plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
    # Plot the 3D surface
    #ax.title.set_text(r"$f(x)=-ln((x-m_1)^2+0.00001)-ln((x-m_2)^2+0.01)$")
    ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm, edgecolor='royalblue')
    ax.set(xlim=x1Lim, ylim=x2Lim, zlim=(np.min(Y), np.max(Y)-1.5))
    ax.set_xlabel('$x_1$', fontsize=25)
    ax.set_ylabel('$x_2$', fontsize=25)
    ax.tick_params(labelsize=20)
    #ax.text(-1.2, -1, 16.6, "$\mathcal{L}(x_1,x_2)=8-\cos(10x_1)-\cos(10x_2)-5x_1^2-5x_2^2$",
    #        color='black', size=20)
    plt.savefig("results/escape-2D-objective.pdf", bbox_inches='tight')
    plt.show()



#plotting Trajectories (in original line 504-584)
def plotTrajectory(rl_trajectory, ga_trajectory,step_num, path):
    
    fig, ax = plt.subplots(figsize=(8,8))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
    plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'

    
    X1, X2, Y=create2Dfunction(x1Lim=(-2, 2), x2Lim=(-1, 3), N=200)
    # CS = ax.contour(X1, X2, Y)
    CS = plt.contour(X1, X2, Y,levels= -np.logspace(-1, 3, 20)[::-1], cmap=plt.get_cmap('jet_r'))
    
    ax.clabel(CS, inline=True, fontsize=20,fmt='%1.1f',
            colors=('r', 'red', 'blue', (1, 1, 0), '#afeeee', '0.5')
            )

    #Plot the REINFORCE-OPT trajectory
    i = 0
    plt.arrow(x=rl_trajectory[i][0],
            y=rl_trajectory[i][1],
            dx=rl_trajectory[i+1][0]-rl_trajectory[i][0],
            dy=rl_trajectory[i+1][1]-rl_trajectory[i][1],
            color='r',linestyle='--',width=0.008, label='REINFORCE-OPT')

    for i in range(0, len(rl_trajectory)-1):   
        plt.arrow(x=rl_trajectory[i][0],
                y=rl_trajectory[i][1],
                dx=rl_trajectory[i+1][0]-rl_trajectory[i][0],
                dy=rl_trajectory[i+1][1]-rl_trajectory[i][1],
                color='r',linestyle='--',width=0.008)

    #Plot the gradient descent trajectory
    i = 0
    plt.arrow(x=ga_trajectory[i][0],
            y=ga_trajectory[i][1],
            dx=ga_trajectory[i+1][0]-ga_trajectory[i][0],
            dy=ga_trajectory[i+1][1]-ga_trajectory[i][1],
            color='b',linestyle='--',width=0.006, label='Gradient Ascent')

    for i in range(0, len(ga_trajectory)-1):   
        plt.arrow(x=ga_trajectory[i][0],
                y=ga_trajectory[i][1],
                dx=ga_trajectory[i+1][0]-ga_trajectory[i][0],
                dy=ga_trajectory[i+1][1]-ga_trajectory[i][1],
                color='b',linestyle='--',width=0.006)

    plt.plot(0,0,marker='o',color='black',markersize=12) 
    plt.text(0, 0, s='Global Max',size=20)
    plt.tick_params(size=8)
    plt.xlabel('$x_1$',size=25)
    plt.ylabel('$x_2$',size=25)
    plt.legend(loc='upper right',fontsize=20)   
    #plt.title(r"Trajectory in the $\mathbf{x}$-space",size=15)
    plt.savefig(path+"/escape-2D-traj1.eps", bbox_inches='tight')
    plt.savefig(path+"/escape-2D-traj1.pdf", bbox_inches='tight')
    plt.savefig(path+"/escape-2D-traj1.png", bbox_inches='tight')
    
    plt.close()


    ########################################################### - Figure 2b
    rl_fitness_traj = []
    ga_fitness_traj = []
    for i in range(step_num):
        rl_fitness_traj.append(fun.f(rl_trajectory[i]))
        
    for i in range(step_num):
        ga_fitness_traj.append(fun.f(ga_trajectory[i]))
        
    plt.figure(figsize=(8,6))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
    plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'

    plt.plot(range(1,step_num+1),rl_fitness_traj,color='red',marker='o',linestyle='--',label='REINFORCE-OPT')
    plt.plot(range(1,step_num+1),ga_fitness_traj,color='blue',marker='x',linestyle='--',label='Gradient Ascent')
    plt.xlabel('$t$',size=28)
    plt.gca().set_xscale('log')
    plt.xlim([0,40]) 
    plt.ylabel('$\mathcal{L}(x_t)$',size=28)
    plt.tick_params(labelsize=20)
    plt.legend(loc='upper left',fontsize=20)
    #plt.title('$\mathcal{L}(x_t)$ Trajectory - The 2D Case',size=25)
    plt.savefig(path+"/fitness-traj-2D.eps", bbox_inches='tight')
    plt.savefig(path+"/fitness-traj-2D.pdf", bbox_inches='tight')
    plt.savefig(path+"/fitness-traj-2D.png", bbox_inches='tight')
    
    plt.close()




if __name__ =='__main__':
    # x = np.array([1,2])
    # plotFunction2D(x1Lim=(-1.0, 1.0), x2Lim=(-1.0, 1.0))
    
    # x = np.linspace(-1, 1, 400)
    # y = np.linspace(-1, 1, 400)
    # X, Y = np.meshgrid(x, y)

    # # Evaluate fun
    # # ction over grid
    # print(np.hstack((X, Y)))
    # Z = fun.f(np.hstack((X, Y)))
    # print(np.stack([X, Y]))
    X, Y,Z=create2Dfunction(x1Lim=(-1.0, 1.0), x2Lim=(-1.0, 1.0), N=200)
    # Plot contour of given function
    plt.figure(figsize=(8, 6))
    # cp = plt.contour(X, Y, Z,levels= -np.logspace(-1, 3, 20)[::-1], cmap=plt.get_cmap('jet_r'))
    cp = plt.contour(X, Y, Z)
    plt.clabel(cp, inline=True, fontsize=8)
    plt.title("Contour Plot of Rosenbrock Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(1, 1, 'ro')  # global minimum
    plt.grid(True)
    plt.colorbar(cp, label="f(x, y)")
    plt.savefig("ccRosenbrockFunction-2D.eps", bbox_inches='tight')
    plt.savefig("ccRosenbrockFunction-2D.pdf", bbox_inches='tight')
    plt.savefig("ccRosenbrockFunction-2D.png", bbox_inches='tight')
    plt.show()

    # a=fun.f(x)
    # print(a)
    
        ############### -- The objective function. In Figure 3a, f(x) is denoted as \mathcal{L}(x).
  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    # ######################### - Plot the Objective Function in Figure 3
    # x1_min = -1.0
    # x2_min = -1.0
    # x1_max = 1.0
    # x2_max = 1.0
    # x_num = 100

    # X1 = np.linspace(x1_min,x1_max,x_num)
    # X2 = np.linspace(x2_min,x2_max,x_num)
    # X1, X2 = np.meshgrid(X1, X2)
    # Y = np.zeros((x_num,x_num))
    # for i in range(x_num):
    #     for j in range(x_num):
    #         Y[i,j] = f( x=np.array([X1[i,j],X2[i,j]]) )

    # ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['mathtext.fontset'] = 'custom'
    # plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
    # plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
    # # Plot the 3D surface
    # #ax.title.set_text(r"$f(x)=-ln((x-m_1)^2+0.00001)-ln((x-m_2)^2+0.01)$")
    # ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm, edgecolor='royalblue')
    # ax.set(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max), zlim=(np.min(Y), np.max(Y)-1.5))
    # ax.set_xlabel('$x_1$', fontsize=25)
    # ax.set_ylabel('$x_2$', fontsize=25)
    # ax.tick_params(labelsize=20)
    # #ax.text(-1.2, -1, 16.6, "$\mathcal{L}(x_1,x_2)=8-\cos(10x_1)-\cos(10x_2)-5x_1^2-5x_2^2$",
    # #        color='black', size=20)
    # plt.show()
    # plt.savefig("escape-2D-objective2d.png", bbox_inches='tight', transparent=True)
    # # plt.show()