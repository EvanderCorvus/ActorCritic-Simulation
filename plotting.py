import matplotlib.pyplot as plt
import numpy as np

def U(U0,x,y):
    bool = (x**2+y**2)**0.5>0.5
    P = 16*U0*(x**2+y**2-0.25)**2
    P[bool] = 0
    return P

def plot_potential(U0):
    plt.xlim(-0.8,0.8)
    plt.ylim(-0.8,0.8)
    plt.scatter(-0.5,0,c='black',label='start',marker='D')    
    plt.scatter(0.5,0,c='black',label='goal',marker='x')


    X1,Y1 = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))

    Potential = U(U0,X1,Y1)/U0
    # _,forcefield = force(X1,Y1)
    plt.imshow(Potential,cmap = 'Greys',extent=[-1,1,-1,1],origin='lower')
    colorbar = plt.colorbar()
    colorbar.set_label(r'$U/U_0$',labelpad=10,fontsize = 20)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

def plot_epoch( x_paths, y_paths, max_steps, epochs, dt,U0):
    episode_plots = np.linspace(epochs//5,epochs,5) 
    for ep in range(len(x_paths)):
        x = x_paths[ep]
        y = y_paths[ep]

        if x.shape[0] == max_steps+1: T = r'$\infty$'
        else: T = str(np.round(len(x)*dt,2))
        path_label = 'T(' + str(int(episode_plots[ep])) +') = '+ T

        plt.plot(x,y,label=path_label)
        # plt.scatter(x,y,label=path_label, marker='.')
    plot_potential(U0)
    plt.legend(bbox_to_anchor=(1,-0.1),ncol=2)
    plt.show()

def plot_simulation_data(total_mean_losses, mse_mean_losses, kl_mean_losses,max_episodes, char_length):
    fig, ax = plt.subplots()
    X = np.linspace(3,max_episodes,max_episodes-2)
    ax.plot(X, total_mean_losses[2:], label = 'Total Loss', c='black')
    ax.plot(X, mse_mean_losses[2:], label = 'MSE Loss')
    ax.plot(X, kl_mean_losses[2:], label = 'KL Loss')

    ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title(r'$\sqrt{D_RL/v_0}$ = '+ str(char_length))
    plt.show()
