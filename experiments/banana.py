# -----------------------------------------------------------------
# This script belongs to the ModularGP repo
# "Modular Gaussian Processes for Transfer Learning" @ NeurIPS 2021
# Copyright (c) 2021 Pablo Moreno-Munoz
# -----------------------------------------------------------------
#
#
# Pablo Moreno-Munoz (pabmo@dtu.dk)
# Section for Cognitive Systems
# Technical University of Denmark (DTU)
# October 2021

# -----------------------------------------------------------------
# Experiment -- Banana Classification
# -----------------------------------------------------------------

from kernels.rbf import RBF
from likelihoods.gaussian import Gaussian
from likelihoods.bernoulli import Bernoulli
from models.svgp import SVGP
from models.ensemblegp import EnsembleGP
from optimization.algorithms import vem_algorithm, ensemble_vem, ensemble_vem_parallel
from optimization.algorithms import AlgorithmVEM
from sklearn.model_selection import train_test_split

import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# COOLORS.CO palettes
color_palette_1 = ['#335c67','#fff3b0','#e09f3e','#9e2a2b','#540b0e']
color_palette_2 = ['#177e89','#084c61','#db3a34','#ef8354','#323031']
color_palette_3 = ['#bce784','#5dd39e','#348aa7','#525274','#513b56']
color_palette_4 = ['#002642','#840032','#e59500','#e5dada','#02040e']
color_palette_5 = ['#202c39','#283845','#b8b08d','#f2d449','#f29559']
color_palette_6 = ['#21295c','#1b3b6f','#065a82','#1c7293','#9eb3c2']
color_palette_7 = ['#f7b267','#f79d65','#f4845f','#f27059','#f25c54']

color_palette = color_palette_6
color_0 = color_palette_6[0]
color_1 = color_palette_6[4]

# Load Data --
data = sio.loadmat('../data/banana.mat')
y_banana = data['banana_Y']
x_banana = data['banana_X']

trials = 10
nlpd_metrics = np.zeros((1,trials))

plot_local = False
plot_ensemble = False
save = False

for trial in range(trials):
    print('TRIAL = ' + str(trial) + '/' + str(trials))
    x, x_test, y, y_test = train_test_split(x_banana, y_banana, test_size=0.33, random_state=42)

    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Sorting wrt first input dimension
    y = y[x[:,0].argsort()]
    x = x[x[:,0].argsort()]

    # plot limits
    max_x = x[:,0].max()
    max_y = x[:,1].max()
    min_x = x[:,0].min()
    min_y = x[:,1].min()

    # Division into 4 regions
    x_1 = torch.from_numpy(x[(x[:,0]<0.0) & (x[:,1]<0.0),:]).float()
    y_1 = torch.from_numpy(y[(x[:,0]<0.0) & (x[:,1]<0.0),:]).float()

    x_2 = torch.from_numpy(x[(x[:,0]>0.0) & (x[:,1]<0.0),:]).float()
    y_2 = torch.from_numpy(y[(x[:,0]>0.0) & (x[:,1]<0.0),:]).float()

    x_3 = torch.from_numpy(x[(x[:,0]>0.0) & (x[:,1]>0.0),:]).float()
    y_3 = torch.from_numpy(y[(x[:,0]>0.0) & (x[:,1]>0.0),:]).float()

    x_4 = torch.from_numpy(x[(x[:,0]<0.0) & (x[:,1]>0.0),:]).float()
    y_4 = torch.from_numpy(y[(x[:,0]<0.0) & (x[:,1]>0.0),:]).float()

    # All tasks
    x_tasks = [x_1, x_2, x_3, x_4]
    y_tasks = [y_1, y_2, y_3, y_4]

    K = len(x_tasks)
    sigmoid = torch.nn.Sigmoid()

    M_k = 3         # inducing points per side
    N_test = 80     # test points per side

    ###########################
    #                         #
    #    DISTRIBUTED TASKS    #
    #                         #
    ###########################

    models = []
    for k, x_k in enumerate(x_tasks):

        print('-                             -')
        print('----- TASK k=' + str(k + 1) + ' ------')
        print('-                             -')

        y_k = y_tasks[k]
        kernel_k = RBF()
        likelihood_k = Bernoulli()
        model_k = SVGP(kernel_k, likelihood_k, M_k**2, input_dim=2)

        # initial grid of inducing-points
        mx = torch.mean(x_k[:, 0])
        my = torch.mean(x_k[:, 1])
        vx = torch.var(x_k[:, 0])
        vy = torch.var(x_k[:, 1])

        zy = np.linspace(my - 3*vy, my + 3*vy, M_k)
        zx = np.linspace(mx - 3*vx, mx + 3*vx, M_k)
        ZX, ZY = np.meshgrid(zx, zy)
        ZX = ZX.reshape(M_k ** 2, 1)
        ZY = ZY.reshape(M_k ** 2, 1)
        Z = np.hstack((ZX, ZY))
        z_k = torch.from_numpy(Z).float()

        model_k.z = torch.nn.Parameter(z_k, requires_grad=True)
        vem_algorithm = AlgorithmVEM(model_k, x_k, y_k, iters=7)

        vem_algorithm.ve_its = 20
        vem_algorithm.vm_its = 10
        vem_algorithm.lr_m = 1e-3
        vem_algorithm.lr_L = 1e-6
        vem_algorithm.lr_hyp = 1e-6
        vem_algorithm.lr_z = 1e-4

        vem_algorithm.fit()
        models.append(model_k)

        # NLPD -- Metrics
        nlpd = model_k.nlpd(x_test, y_test)

        print('Local Model ('+str(k+1)+')- NLPD: ', nlpd)
        print(' ')

        if plot_local:

            min_tx = x[:,0].min() - 0.15
            min_ty = x[:,1].min() - 0.15
            max_tx = x[:,0].max() + 0.15
            max_ty = x[:,1].max() + 0.15

            ty = np.linspace(min_ty, max_ty, N_test)
            tx = np.linspace(min_tx, max_tx, N_test)
            TX_grid, TY_grid = np.meshgrid(tx, ty)
            TX = TX_grid.reshape(N_test ** 2, 1)
            TY = TY_grid.reshape(N_test ** 2, 1)
            X_test = np.hstack((TX, TY))
            x_test = torch.from_numpy(X_test).float()

            gp, gp_upper, gp_lower = model_k.predictive(x_test)
            gp = sigmoid(torch.from_numpy(gp))

            # Plot
            plt.figure(figsize=(7, 6))
            ax = plt.axes()
            plt.plot(x_k[y_k[:, 0] == 0, 0], x_k[y_k[:, 0] == 0, 1], 'o', color=color_1, alpha=0.5)
            plt.plot(x_k[y_k[:, 0] == 1, 0], x_k[y_k[:, 0] == 1, 1], 'o', color=color_0, alpha=0.5)
            plt.plot(model_k.z[:,0].detach(), model_k.z[:,1].detach(), 'kx',  ms=10.0, mew=2.0)
            cs = ax.contour(TX_grid, TY_grid, np.reshape(gp, (N_test, N_test)), linewidths=3, colors='k',
                        levels=[0.25, 0.5, 0.75], zorder=10)
            ax.clabel(cs, inline=1, fontsize=14, fmt='%1.1f')

            plt.title(r'Banana Recyclable GP - '+ str(k + 1) )
            plt.xlabel(r'$x_1$ input')
            plt.ylabel(r'$x_2$ input')
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)

            if save:
                plt.savefig(fname='./figs/banana/banana_task_' + str(k + 1) + '.pdf', format='pdf')

            #plt.show()
            plt.close()

    ###########################
    #                         #
    #   ENSEMBLE INFERENCE    #
    #                         #
    ###########################

    print('-                   -')
    print('----- ENSEMBLE ------')
    print('-                   -')

    M_e = 5
    kernel = RBF()
    likelihood = Bernoulli()
    model_e = EnsembleGP(kernel, likelihood, models, M_e**2, input_dim=2)

    # initial grid of inducing-points
    mx = np.mean(x[:, 0])
    my = np.mean(x[:, 1])
    vx = np.var(x[:, 0])
    vy = np.var(x[:, 1])

    zy = np.linspace(my - 1.5*vy, my + 1.5*vy, M_e)
    zx = np.linspace(mx - 1.5*vx, mx + 1.5*vx, M_e)
    ZX, ZY = np.meshgrid(zx, zy)
    ZX = ZX.reshape(M_e ** 2, 1)
    ZY = ZY.reshape(M_e ** 2, 1)
    Z = np.hstack((ZX, ZY))
    z_e = torch.from_numpy(Z).float()

    model_e.z = torch.nn.Parameter(z_e, requires_grad=True)
    vem_algorithm = AlgorithmVEM(model_e, config='ensemble', iters=20)

    vem_algorithm.ve_its = 20
    vem_algorithm.vm_its = 10
    vem_algorithm.lr_m = 1e-3
    vem_algorithm.lr_L = 1e-5
    vem_algorithm.lr_hyp = 1e-6
    vem_algorithm.lr_z = 1e-5

    vem_algorithm.fit()

    # NLPD -- Metrics
    nlpd = model_e.nlpd(x_test, y_test)

    nlpd_metrics[0, trial] = nlpd

    print('Banana Ensemble NLPD: ', nlpd)
    print(' ')

    if plot_ensemble:

        min_tx = x[:,0].min() - 0.15
        min_ty = x[:,1].min() - 0.15
        max_tx = x[:,0].max() + 0.15
        max_ty = x[:,1].max() + 0.15

        ty = np.linspace(min_ty, max_ty, N_test)
        tx = np.linspace(min_tx, max_tx, N_test)
        TX_grid, TY_grid = np.meshgrid(tx, ty)
        TX = TX_grid.reshape(N_test ** 2, 1)
        TY = TY_grid.reshape(N_test ** 2, 1)
        X_test = np.hstack((TX, TY))
        x_test = torch.from_numpy(X_test).float()

        gp, _, _ = model_e.predictive(x_test)
        gp = sigmoid(torch.from_numpy(gp))

        # Plot
        plt.figure(figsize=(7, 6))
        ax = plt.axes()
        plt.plot(x[y[:, 0] == 0, 0], x[y[:, 0] == 0, 1], 'o', color=color_1, alpha=0.5)
        plt.plot(x[y[:, 0] == 1, 0], x[y[:, 0] == 1, 1], 'o', color=color_0, alpha=0.5)
        plt.plot(model_e.z[:,0].detach(), model_e.z[:,1].detach(), 'kx', ms=10.0, mew=2.0)
        cs = ax.contour(TX_grid, TY_grid, np.reshape(gp, (N_test, N_test)), linewidths=3, colors='k',
                    levels=[0.25, 0.5, 0.75], zorder=10)
        ax.clabel(cs, inline=1, fontsize=14, fmt='%1.1f')

        plt.title(r'Banana GP Ensemble')
        plt.xlabel(r'$x_1$ input')
        plt.ylabel(r'$x_2$ input')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)

        if save:
            plt.savefig(fname='./figs/banana/banana_task_ensemble.pdf', format='pdf')

        plt.show()
        #plt.close()

