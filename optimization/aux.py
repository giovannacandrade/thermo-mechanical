from matplotlib import pyplot as plt
import numpy as np

def plot_results(J, comp, vol, It, lag_it, rho_it, dJ, volfrac, zeta):
    fig, ax1 = plt.subplots()
    plt.grid(True)
    color = 'tab:blue'
    ax1.plot(comp[0:It],'--', lw = 1.3, label = 'Compliance', color=color)
    ax1.plot(J[0:It],'.-',lw = 0.5, label = 'Objective function', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.legend(fontsize = 18,loc = 'upper right')
    ax2 = ax1.twinx() 
    color = 'darkgreen'
    ax2.set_ylabel('Fraction of volume', color=color,  rotation=270, labelpad=20, fontsize = 20); 
    ax2.plot(vol[0:It],'.-', lw = 0.5, label = 'Fraction of volume', color=color)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel('Iterations', fontsize = 20); 
    plt.tight_layout()
    plt.savefig('compJ.pdf')
    
    fig, ax1 = plt.subplots()
    plt.grid(True)
    color = 'tab:red'
    ax1.set_xlabel('Iterations',  fontsize = 20); ax1.set_ylabel('Lagrange multiplier', color=color, fontsize = 20)
    ax1.plot(lag_it[0:It],'.-', lw = 0.5, label = 'r$\Lambda$', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2 = ax1.twinx() 
    color = 'tab:purple'
    ax2.set_ylabel('Penalization parameter', color=color,  rotation=270, labelpad=20, fontsize = 20); 
    ax2.plot(rho_it[0:It],'.-', lw = 0.5, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig('lag.pdf')  
    c = zeta*(np.array(vol) - volfrac)
    aux = np.zeros(It)
    for i in range(1,It):
        if c[i] >= 0: aux[i] = 2
        elif c[i] > - lag_it[i-1]/rho_it[i-1] and c[i]<0: aux[i] = 1
        else: aux[i] = 0
    x = np.arange(-5,It+5)
    f, ax = plt.subplots()
    plt.plot(x,2*np.ones(np.size(x)), '-', label = 'c(x) > 0'); plt.plot(x, 1*np.ones(np.size(x)), '-', label = ' 0 > c(x) > -'+r'$\mu/\rho$');
    plt.plot(x,0*np.ones(np.size(x)), '-', label = 'c(x) < -'+r'$\mu/\rho$', color = 'tab:red');
    plt.plot(aux[1:It],'o',color = 'k'); 
    plt.legend();
    ax.set_ylim([-1,4]);   
    ax.grid(axis='x')
    plt.yticks([])
    plt.savefig('lagAum.pdf')