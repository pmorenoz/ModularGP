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
# Experiment -- Baselines (Million)
# -----------------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt

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

color_palette = color_palette_2

from kernels.rbf import RBF
from likelihoods.gaussian import Gaussian
from models.svgp import SVGP
from models.ensemblegp import EnsembleGP
from baselines.distgp import DistGP
from baselines.poegp import PoeGP
from baselines.gpoegp import GenPoeGP
from baselines.bcm import BayesianCM
from baselines.rbcm import RobustBayesianCM
from baselines.dvigp import DVIGP
from optimization.algorithms import AlgorithmVEM
from optimization.algorithms import GPR_Optimizer
from util import smooth_function

experiment = '1m'

N_k = 50 # 200
M_k = 3
M_e = 35

T = 50
tasks = 50
layer_1_merge = 2 # 10
layer_2_merge = 2 # 10
trials = 1
node_overlapping = 1

plot_layer_0 = False
plot_layer_1 = True
plot_layer_2 = True
plot_ensemble = True
save = True

recy_metrics = np.zeros((3,trials))
poe_metrics = np.zeros((3,trials))
gpoe_metrics = np.zeros((3,trials))
bcm_metrics = np.zeros((3,trials))
rbcm_metrics = np.zeros((3,trials))

N_test = 400
min_x = 0.0
max_x = T * 0.1
segment_x = (max_x - min_x) / tasks
x_test = torch.linspace(min_x - 0.5, max_x + 0.5, N_test)[:, None]
f_test = smooth_function(x_test)
y_test = f_test + 2.0 * torch.randn(N_test, 1)

for trial in range(trials):
    print('TRIAL = '+str(trial+1)+'/'+str(trials))
    layer_2 = []
    layer_2_poe_gpm = []  # POE GPs (predictive)
    layer_2_poe_gpv = []  # POE GPs (predictive)
    layer_2_gpoe_gpm = []  # GPOE GPs (predictive)
    layer_2_gpoe_gpv = []  # GPOE GPs (predictive)
    layer_2_bcm_gpm = []  # BCM GPs (predictive)
    layer_2_bcm_gpv = []  # BCM GPs (predictive)
    layer_2_rbcm_gpm = []  # rBCM GPs (predictive)
    layer_2_rbcm_gpv = []  # rBCM GPs (predictive)

    for j in range(layer_2_merge):
        print('LAYER-2 = ' + str(j+1) + '/' + str(layer_2_merge))
        layer_1 = []
        layer_1_poe_gpm = []  # POE GPs (predictive)
        layer_1_poe_gpv = []  # POE GPs (predictive)
        layer_1_gpoe_gpm = []  # GPOE GPs (predictive)
        layer_1_gpoe_gpv = []  # GPOE GPs (predictive)
        layer_1_bcm_gpm = []  # BCM GPs (predictive)
        layer_1_bcm_gpv = []  # BCM GPs (predictive)
        layer_1_rbcm_gpm = []  # rBCM GPs (predictive)
        layer_1_rbcm_gpv = []  # rBCM GPs (predictive)

        for m in range(layer_1_merge):
            print('LAYER-1 = ' + str(m+1) + '/' + str(layer_1_merge))

            ###########################
            #        LAYER 0          #
            #      ___________        #
            #      DISTRIBUTED        #
            ###########################

            x_tasks = []
            y_tasks = []

            # SYNTHETIC DATA
            for n in range(node_overlapping):
                for k in range(T):
                    x_k = ((min_x + (k * segment_x)) - (min_x + ((k + 1) * segment_x))) * torch.rand(N_k, 1) + (
                                min_x + ((k + 1) * segment_x))
                    x_k, _ = torch.sort(x_k, dim=0)
                    y_k = smooth_function(x_k) + 2.0 * torch.randn(N_k, 1)
                    x_tasks.append(x_k)
                    y_tasks.append(y_k)

            tasks = T * node_overlapping

            layer_0 = []            # recyclable GPs
            layer_0_dist = []       # distributed GPs (models)
            layer_0_dist_gpm = []   # distributed GPs (predictive)
            layer_0_dist_gpv = []  # distributed GPs (predictive)
            for k, x_k in enumerate(x_tasks):
                print(' ')
                print('TRIAL   = ' + str(trial + 1) + '/' + str(trials))
                print('LAYER-0 = ' + str(k+1) + '/' + str(T*node_overlapping))
                print('LAYER-1 = ' + str(m+1) + '/' + str(layer_1_merge))
                print('LAYER-2 = ' + str(j+1) + '/' + str(layer_2_merge))
                print('\                             -')
                print(' ---- TASK k=' + str(k + 1) + ' ------')
                print('/                             -')
                print(' ')
                ######################################################
                # 1. RECYCLABLE GP
                ######################################################
                kernel_k = RBF()
                likelihood_k = Gaussian(fit_noise=False)
                model_k = SVGP(kernel_k, likelihood_k, M_k)

                z_k_min = min_x + ((k % T) * segment_x)
                z_k_max = min_x + (((k % T) + 1) * segment_x)
                model_k.z = torch.nn.Parameter(torch.linspace(z_k_min, z_k_max, M_k)[:, None], requires_grad=True)
                vem_algorithm = AlgorithmVEM(model_k, x_k, y_tasks[k], iters=15)

                vem_algorithm.ve_its = 20
                vem_algorithm.vm_its = 10
                vem_algorithm.lr_m = 1e-6
                vem_algorithm.lr_L = 1e-10
                vem_algorithm.lr_hyp = 1e-10
                vem_algorithm.lr_z = 1e-10

                vem_algorithm.fit()
                layer_0.append(model_k)

                ######################################################
                # 2. DISTRIBUTED GP (FOR BCM, RBCM, POE & GPOE)
                ######################################################

                kernel_j = RBF()
                likelihood_j = Gaussian(fit_noise=True)
                model_j = DistGP(kernel_j, likelihood_j)
                GPR_Optimizer(model_j, x_k, y_tasks[k])

                dis_gp_m, dis_gp_v = model_j.predictive(x_k, y_tasks[k], x_test)
                layer_0_dist.append(model_j)
                layer_0_dist_gpm.append(dis_gp_m)
                layer_0_dist_gpv.append(dis_gp_v)

                if plot_layer_0:
                    gp, gp_upper, gp_lower = model_k.predictive(x_test)

                    disgp = dis_gp_m.detach().numpy()
                    disgp_upper = (dis_gp_m + 2 * torch.sqrt(dis_gp_v)).detach().numpy() + 2 * model_j.likelihood.sigma.detach().numpy()
                    disgp_lower = (dis_gp_m - 2 * torch.sqrt(dis_gp_v)).detach().numpy() - 2 * model_j.likelihood.sigma.detach().numpy()

                    plt.figure(figsize=(12, 4))
                    plt.plot(x_k, y_tasks[k], ls='-', color=color_palette[k % len(color_palette)], markersize=2.5, markeredgewidth=0.75)
                    plt.plot(layer_0[k].z.detach(), -20.0 * torch.ones(M_k, 1), color=color_palette[k % len(color_palette)],linestyle='', marker='.', markersize=5)

                    plt.plot(x_test, gp, 'k-', linewidth=1.5)
                    plt.plot(x_test, gp_upper, 'k-', linewidth=2.5)
                    plt.plot(x_test, gp_lower, 'k-', linewidth=2.5)

                    plt.plot(x_test, disgp, 'b-', linewidth=1.5)
                    plt.plot(x_test, disgp_upper, 'b-', linewidth=2.5)
                    plt.plot(x_test, disgp_lower, 'b-', linewidth=2.5)

                    plt.title(r'Variational Sparse GP -- (task=' + str(k + 1) + ')')
                    plt.xlabel(r'Input, $x$')
                    plt.ylabel(r'Output, $y$')
                    plt.xlim(min_x - 0.5, max_x + 0.5)
                    plt.ylim(-22.0, 22.0)

                    plt.show()

            ###########################
            #        LAYER 0          #
            #        ________         #
            #        ENSEMBLE         #
            ###########################

            print(' ')
            print('TRIAL   = ' + str(trial + 1) + '/' + str(trials))
            print('LAYER-0 = ' + str(k + 1) + '/' + str(T * node_overlapping))
            print('LAYER-1 = ' + str(m + 1) + '/' + str(layer_1_merge))
            print('LAYER-2 = ' + str(j + 1) + '/' + str(layer_2_merge))
            print('\                             -')
            print(' ------ ENSEMBLE LAYER 0 ------')
            print('/                             -')
            print(' ')

            ######################################################
            # 1. ENSEMBLE RECYCLABLE GP
            ######################################################

            kernel = RBF()
            likelihood = Gaussian(fit_noise=False)
            e_layer_0 = EnsembleGP(kernel, likelihood, layer_0, M_e)
            e_layer_0.z = torch.nn.Parameter(torch.linspace(min_x, max_x, M_e)[:,None], requires_grad=True)
            vem_algorithm = AlgorithmVEM(e_layer_0, config='ensemble', iters=10)

            vem_algorithm.ve_its = 30
            vem_algorithm.vm_its = 10
            vem_algorithm.lr_m = 1e-3
            vem_algorithm.lr_L = 1e-6
            vem_algorithm.lr_hyp = 1e-8
            vem_algorithm.lr_z = 1e-8

            vem_algorithm.fit()
            layer_1.append(e_layer_0)

            #########################################################
            # 2. ENSEMBLE DISTRIBUTED GP (FOR BCM, RBCM, POE & GPOE)
            #########################################################
            # A. POE  _________//
            # B. GPOE _________//
            # C. BCM  _________//
            # D. RBCM _________//

            poe_model = PoeGP(models=layer_0_dist)
            gpoe_model = GenPoeGP(models=layer_0_dist)
            bcm_model = BayesianCM(models=layer_0_dist)
            rbcm_model = RobustBayesianCM(models=layer_0_dist)

            poe_m, poe_v = poe_model.predictive_layer(layer_0_dist_gpm, layer_0_dist_gpv, x_test)
            gpoe_m, gpoe_v = gpoe_model.predictive_layer(layer_0_dist_gpm, layer_0_dist_gpv, x_test)
            bcm_m, bcm_v = bcm_model.predictive_layer(layer_0_dist_gpm, layer_0_dist_gpv, x_test)
            rbcm_m, rbcm_v = rbcm_model.predictive_layer(layer_0_dist_gpm, layer_0_dist_gpv, x_test)

            layer_1_poe_gpm.append(poe_m)
            layer_1_poe_gpv.append(poe_v)
            layer_1_gpoe_gpm.append(gpoe_m)
            layer_1_gpoe_gpv.append(gpoe_v)
            layer_1_bcm_gpm.append(bcm_m)
            layer_1_bcm_gpv.append(bcm_v)
            layer_1_rbcm_gpm.append(rbcm_m)
            layer_1_rbcm_gpv.append(rbcm_v)

            if plot_layer_1:
                gp, gp_upper, gp_lower = e_layer_0.predictive(x_test)

                plt.figure(figsize=(12, 4))
                for k in range(T):
                    plt.plot(x_tasks[k], y_tasks[k], ls='-', color=color_palette[k % len(color_palette)],markersize=2.5, markeredgewidth=0.75)
                    plt.plot(layer_0[k].z.detach(), -20.0 * torch.ones(M_k, 1), color=color_palette[k % len(color_palette)], linestyle='', marker='.', markersize=5)

                plt.plot(e_layer_0.z.detach(), -20.0 * torch.ones(M_e, 1), color='r', linestyle='', marker='x',markersize=5,markeredgewidth=1.0)
                plt.plot(x_test, gp, 'k-', linewidth=1.5)
                plt.plot(x_test, gp_upper, 'k-', linewidth=2.5)
                plt.plot(x_test, gp_lower, 'k-', linewidth=2.5)

                poe = poe_m.detach().numpy()
                poe_upper = (poe_m + 2 * torch.sqrt(poe_v)).detach().numpy()  # + 2*model_2.likelihood.sigma.detach().numpy()
                poe_lower = (poe_m - 2 * torch.sqrt(poe_v)).detach().numpy()  # - 2*model_2.likelihood.sigma.detach().numpy()

                plt.plot(x_test, poe, 'g-', linewidth=1.5)
                plt.plot(x_test, poe_upper, 'g-', linewidth=2.5)
                plt.plot(x_test, poe_lower, 'g-', linewidth=2.5)

                gpoe = gpoe_m.detach().numpy()
                gpoe_upper = (gpoe_m + 2 * torch.sqrt(gpoe_v)).detach().numpy()  # + 2*model_2.likelihood.sigma.detach().numpy()
                gpoe_lower = (gpoe_m - 2 * torch.sqrt(gpoe_v)).detach().numpy()  # - 2*model_2.likelihood.sigma.detach().numpy()

                plt.plot(x_test, gpoe, 'm-', linewidth=1.5)
                plt.plot(x_test, gpoe_upper, 'm-', linewidth=2.5)
                plt.plot(x_test, gpoe_lower, 'm-', linewidth=2.5)

                bcm = bcm_m.detach().numpy()
                bcm_upper = (bcm_m + 2 * torch.sqrt(bcm_v)).detach().numpy()  # + 2*model_2.likelihood.sigma.detach().numpy()
                bcm_lower = (bcm_m - 2 * torch.sqrt(bcm_v)).detach().numpy()  # - 2*model_2.likelihood.sigma.detach().numpy()

                plt.plot(x_test, bcm, 'r-', linewidth=1.5)
                plt.plot(x_test, bcm_upper, 'r-', linewidth=2.5)
                plt.plot(x_test, bcm_lower, 'r-', linewidth=2.5)

                rbcm = rbcm_m.detach().numpy()
                rbcm_upper = (rbcm_m + 2 * torch.sqrt(rbcm_v)).detach().numpy()  # + 2*model_2.likelihood.sigma.detach().numpy()
                rbcm_lower = (rbcm_m - 2 * torch.sqrt(rbcm_v)).detach().numpy()  # - 2*model_2.likelihood.sigma.detach().numpy()

                plt.plot(x_test, rbcm, 'b-', linewidth=1.5)
                plt.plot(x_test, rbcm_upper, 'b-', linewidth=2.5)
                plt.plot(x_test, rbcm_lower, 'b-', linewidth=2.5)

                plt.title(r'Variational Sparse GP -- (task=' + str(k + 1) + ')')
                plt.xlabel(r'Input, $x$')
                plt.ylabel(r'Output, $y$')
                plt.xlim(min_x - 0.5, max_x + 0.5)
                plt.ylim(-22.0, 22.0)

                if save:
                    plt.savefig(fname='./figs/baseline/ensemble_layer1_(' + str(m + 1) + ').pdf',format='pdf')

                plt.show()

        ###########################
        #        LAYER 1         #
        ###########################

        print(' ')
        print('TRIAL   = ' + str(trial + 1) + '/' + str(trials))
        print('LAYER-0 = ' + str(k + 1) + '/' + str(T * node_overlapping))
        print('LAYER-1 = ' + str(m + 1) + '/' + str(layer_1_merge))
        print('LAYER-2 = ' + str(j + 1) + '/' + str(layer_2_merge))
        print('\                             -')
        print(' ------ ENSEMBLE LAYER 1 ------')
        print('/                             -')
        print(' ')

        ######################################################
        # 1. ENSEMBLE RECYCLABLE GP
        ######################################################

        kernel = RBF()
        likelihood = Gaussian(fit_noise=False)
        e_layer_1 = EnsembleGP(kernel, likelihood, layer_1, M_e)
        e_layer_1.z = torch.nn.Parameter(torch.linspace(min_x, max_x, M_e)[:, None], requires_grad=True)
        vem_algorithm = AlgorithmVEM(e_layer_1, config='ensemble', iters=10)

        vem_algorithm.ve_its = 30
        vem_algorithm.vm_its = 10
        vem_algorithm.lr_m = 1e-3
        vem_algorithm.lr_L = 1e-6
        vem_algorithm.lr_hyp = 1e-8
        vem_algorithm.lr_z = 1e-8

        vem_algorithm.fit()
        layer_2.append(e_layer_1)

        #########################################################
        # 2. ENSEMBLE DISTRIBUTED GP (FOR BCM, RBCM, POE & GPOE)
        #########################################################
        # A. POE  _________//
        # B. GPOE _________//
        # C. BCM  _________//
        # D. RBCM _________//

        poe_model = PoeGP(models=layer_0_dist)
        gpoe_model = GenPoeGP(models=layer_0_dist)
        bcm_model = BayesianCM(models=layer_0_dist)
        rbcm_model = RobustBayesianCM(models=layer_0_dist)

        poe_m, poe_v = poe_model.predictive_layer(layer_1_poe_gpm, layer_1_poe_gpv, x_test)
        gpoe_m, gpoe_v = gpoe_model.predictive_layer(layer_1_gpoe_gpm, layer_1_gpoe_gpv, x_test)
        bcm_m, bcm_v = bcm_model.predictive_layer(layer_1_bcm_gpm, layer_1_bcm_gpv, x_test)
        rbcm_m, rbcm_v = rbcm_model.predictive_layer(layer_1_rbcm_gpm, layer_1_rbcm_gpv, x_test)

        layer_2_poe_gpm.append(poe_m)
        layer_2_poe_gpv.append(poe_v)
        layer_2_gpoe_gpm.append(gpoe_m)
        layer_2_gpoe_gpv.append(gpoe_v)
        layer_2_bcm_gpm.append(bcm_m)
        layer_2_bcm_gpv.append(bcm_v)
        layer_2_rbcm_gpm.append(rbcm_m)
        layer_2_rbcm_gpv.append(rbcm_v)

        if plot_layer_2:
            gp, gp_upper, gp_lower = e_layer_1.predictive(x_test)

            plt.figure(figsize=(12, 4))
            for k in range(T):
                plt.plot(x_tasks[k], y_tasks[k], ls='-', color=color_palette[k % len(color_palette)], markersize=2.5,
                         markeredgewidth=0.75)

            plt.plot(e_layer_1.z.detach(), -20.0 * torch.ones(M_e, 1), color='r', linestyle='', marker='x', markersize=5,
                     markeredgewidth=1.0)
            plt.plot(x_test, gp, 'k-', linewidth=1.5)
            plt.plot(x_test, gp_upper, 'k-', linewidth=2.5)
            plt.plot(x_test, gp_lower, 'k-', linewidth=2.5)

            poe = poe_m.detach().numpy()
            poe_upper = (poe_m + 2 * torch.sqrt(poe_v)).detach().numpy()  # + 2*model_2.likelihood.sigma.detach().numpy()
            poe_lower = (poe_m - 2 * torch.sqrt(poe_v)).detach().numpy()  # - 2*model_2.likelihood.sigma.detach().numpy()

            plt.plot(x_test, poe, 'g-', linewidth=1.5)
            plt.plot(x_test, poe_upper, 'g-', linewidth=2.5)
            plt.plot(x_test, poe_lower, 'g-', linewidth=2.5)

            gpoe = gpoe_m.detach().numpy()
            gpoe_upper = (gpoe_m + 2 * torch.sqrt(gpoe_v)).detach().numpy()  # + 2*model_2.likelihood.sigma.detach().numpy()
            gpoe_lower = (gpoe_m - 2 * torch.sqrt(gpoe_v)).detach().numpy()  # - 2*model_2.likelihood.sigma.detach().numpy()

            plt.plot(x_test, gpoe, 'm-', linewidth=1.5)
            plt.plot(x_test, gpoe_upper, 'm-', linewidth=2.5)
            plt.plot(x_test, gpoe_lower, 'm-', linewidth=2.5)

            bcm = bcm_m.detach().numpy()
            bcm_upper = (bcm_m + 2 * torch.sqrt(bcm_v)).detach().numpy()  # + 2*model_2.likelihood.sigma.detach().numpy()
            bcm_lower = (bcm_m - 2 * torch.sqrt(bcm_v)).detach().numpy()  # - 2*model_2.likelihood.sigma.detach().numpy()

            plt.plot(x_test, bcm, 'r-', linewidth=1.5)
            plt.plot(x_test, bcm_upper, 'r-', linewidth=2.5)
            plt.plot(x_test, bcm_lower, 'r-', linewidth=2.5)

            rbcm = rbcm_m.detach().numpy()
            rbcm_upper = (rbcm_m + 2 * torch.sqrt(rbcm_v)).detach().numpy()  # + 2*model_2.likelihood.sigma.detach().numpy()
            rbcm_lower = (rbcm_m - 2 * torch.sqrt(rbcm_v)).detach().numpy()  # - 2*model_2.likelihood.sigma.detach().numpy()

            plt.plot(x_test, rbcm, 'b-', linewidth=1.5)
            plt.plot(x_test, rbcm_upper, 'b-', linewidth=2.5)
            plt.plot(x_test, rbcm_lower, 'b-', linewidth=2.5)

            plt.title(r'Variational Sparse GP -- (task=' + str(k + 1) + ')')
            plt.xlabel(r'Input, $x$')
            plt.ylabel(r'Output, $y$')
            plt.xlim(min_x - 0.5, max_x + 0.5)
            plt.ylim(-22.0, 22.0)

            if save:
                plt.savefig(fname='./figs/baseline/ensemble_layer1_('+str(m+1)+')_layer2_('+str(j+1)+').pdf', format='pdf')

            plt.show()

    ###########################
    #        LAYER 2         #
    ###########################

    print(' ')
    print('TRIAL   = ' + str(trial + 1) + '/' + str(trials))
    print('LAYER-0 = ' + str(k + 1) + '/' + str(T * node_overlapping))
    print('LAYER-1 = ' + str(m + 1) + '/' + str(layer_1_merge))
    print('LAYER-2 = ' + str(j + 1) + '/' + str(layer_2_merge))
    print('\                             -')
    print(' ------  FINAL ENSEMBLE  ------')
    print('/                             -')
    print(' ')

    ######################################################
    # 1. ENSEMBLE RECYCLABLE GP
    ######################################################

    kernel = RBF()
    likelihood = Gaussian(fit_noise=False)
    model_e = EnsembleGP(kernel, likelihood, layer_2, M_e)
    model_e.z = torch.nn.Parameter(torch.linspace(min_x, max_x, M_e)[:,None], requires_grad=True)
    vem_algorithm = AlgorithmVEM(model_e, config='ensemble', iters=10)

    vem_algorithm.ve_its = 30
    vem_algorithm.vm_its = 10
    vem_algorithm.lr_m = 1e-3
    vem_algorithm.lr_L = 1e-6
    vem_algorithm.lr_hyp = 1e-8
    vem_algorithm.lr_z = 1e-8

    vem_algorithm.fit()

    #########################################################
    # 2. ENSEMBLE DISTRIBUTED GP (FOR BCM, RBCM, POE & GPOE)
    #########################################################
    # A. POE  _________//
    # B. GPOE _________//
    # C. BCM  _________//
    # D. RBCM _________//

    poe_model = PoeGP(models=layer_0_dist)
    gpoe_model = GenPoeGP(models=layer_0_dist)
    bcm_model = BayesianCM(models=layer_0_dist)
    rbcm_model = RobustBayesianCM(models=layer_0_dist)

    #########################################################
    # -- METRICS --------------------------------------------
    #########################################################
    # X. Recyclable GP _________//

    nlpd = model_e.nlpd(x_test, y_test)
    rmse = model_e.rmse(x_test, f_test)
    mae = model_e.mae(x_test, f_test)

    recy_metrics[0, trial] = nlpd
    recy_metrics[1, trial] = rmse
    recy_metrics[2, trial] = mae

    print('Recyclable - NLPD: ', nlpd)
    print('Recyclable - RMSE: ', rmse)
    print('Recyclable - MAE: ',  mae)
    print(' ')

    # A. POE  _________//

    nlpd = poe_model.nlpd_layer(layer_2_poe_gpm, layer_2_poe_gpv, x_test, y_test)
    rmse = poe_model.rmse_layer(layer_2_poe_gpm, layer_2_poe_gpv, x_test, f_test)
    mae = poe_model.mae_layer(layer_2_poe_gpm, layer_2_poe_gpv, x_test, f_test)

    poe_metrics[0, trial] = nlpd
    poe_metrics[1, trial] = rmse
    poe_metrics[2, trial] = mae

    print('POE-NLPD: ', nlpd)
    print('POE-RMSE: ', rmse)
    print('POE-MAE: ',  mae)
    print(' ')

    # B. GPOE _________//

    nlpd = gpoe_model.nlpd_layer(layer_2_gpoe_gpm, layer_2_gpoe_gpv, x_test, y_test)
    rmse = gpoe_model.rmse_layer(layer_2_gpoe_gpm, layer_2_gpoe_gpv, x_test, f_test)
    mae = gpoe_model.mae_layer(layer_2_gpoe_gpm, layer_2_gpoe_gpv, x_test, f_test)

    gpoe_metrics[0, trial] = nlpd
    gpoe_metrics[1, trial] = rmse
    gpoe_metrics[2, trial] = mae

    print('GenPOE-NLPD: ', nlpd)
    print('GenPOE-RMSE: ', rmse)
    print('GenPOE-MAE: ',  mae)
    print(' ')

    # C. BCM  _________//

    nlpd = bcm_model.nlpd_layer(layer_2_bcm_gpm, layer_2_bcm_gpv, x_test, y_test)
    rmse = bcm_model.rmse_layer(layer_2_bcm_gpm, layer_2_bcm_gpv, x_test, f_test)
    mae = bcm_model.mae_layer(layer_2_bcm_gpm, layer_2_bcm_gpv, x_test, f_test)

    bcm_metrics[0, trial] = nlpd
    bcm_metrics[1, trial] = rmse
    bcm_metrics[2, trial] = mae

    print('BCM-NLPD: ', nlpd)
    print('BCM-RMSE: ', rmse)
    print('BCM-MAE: ',  mae)
    print(' ')

    # D. RBCM _________//

    nlpd = rbcm_model.nlpd_layer(layer_2_rbcm_gpm, layer_2_rbcm_gpv, x_test, y_test)
    rmse = rbcm_model.rmse_layer(layer_2_rbcm_gpm, layer_2_rbcm_gpv, x_test, f_test)
    mae = rbcm_model.mae_layer(layer_2_rbcm_gpm, layer_2_rbcm_gpv, x_test, f_test)

    rbcm_metrics[0, trial] = nlpd
    rbcm_metrics[1, trial] = rmse
    rbcm_metrics[2, trial] = mae

    print('RBCM-NLPD: ', nlpd)
    print('RBCM-RMSE: ', rmse)
    print('RBCM-MAE: ',  mae)
    print(' ')

    # save to csv file
    np.savetxt('./metrics/recy_metrics_'+ experiment +'.csv', recy_metrics, delimiter=',')
    np.savetxt('./metrics/poe_metrics_' + experiment + '.csv', poe_metrics, delimiter=',')
    np.savetxt('./metrics/gpoe_metrics_' + experiment + '.csv', gpoe_metrics, delimiter=',')
    np.savetxt('./metrics/bcm_metrics_' + experiment + '.csv', bcm_metrics, delimiter=',')
    np.savetxt('./metrics/rbcm_metrics_' + experiment + '.csv', rbcm_metrics, delimiter=',')

    if plot_ensemble:
        gp, gp_upper, gp_lower = model_e.predictive(x_test)

        plt.figure(figsize=(12, 4))
        poe_m, poe_v = poe_model.predictive_layer(layer_2_poe_gpm, layer_2_poe_gpv, x_test)
        gpoe_m, gpoe_v = gpoe_model.predictive_layer(layer_2_gpoe_gpm, layer_2_gpoe_gpv, x_test)
        bcm_m, bcm_v = bcm_model.predictive_layer(layer_2_bcm_gpm, layer_2_bcm_gpv, x_test)
        rbcm_m, rbcm_v = rbcm_model.predictive_layer(layer_2_rbcm_gpm, layer_2_rbcm_gpv, x_test)

        for k in range(T):
            plt.plot(x_tasks[k], y_tasks[k], ls='-', color=color_palette[k % len(color_palette)], markersize=2.5,
                     markeredgewidth=0.75)

        plt.plot(model_e.z.detach(), -20.0 * torch.ones(M_e, 1), color='r', linestyle='', marker='x', markersize=5,
                 markeredgewidth=1.0)
        plt.plot(x_test, gp, 'k-', linewidth=1.5)
        plt.plot(x_test, gp_upper, 'k-', linewidth=2.5)
        plt.plot(x_test, gp_lower, 'k-', linewidth=2.5)

        poe = poe_m.detach().numpy()
        poe_upper = (poe_m + 2 * torch.sqrt(poe_v)).detach().numpy()  # + 2*model_2.likelihood.sigma.detach().numpy()
        poe_lower = (poe_m - 2 * torch.sqrt(poe_v)).detach().numpy()  # - 2*model_2.likelihood.sigma.detach().numpy()

        plt.plot(x_test, poe, 'g-', linewidth=1.5)
        plt.plot(x_test, poe_upper, 'g-', linewidth=2.5)
        plt.plot(x_test, poe_lower, 'g-', linewidth=2.5)

        gpoe = gpoe_m.detach().numpy()
        gpoe_upper = (gpoe_m + 2 * torch.sqrt(gpoe_v)).detach().numpy()  # + 2*model_2.likelihood.sigma.detach().numpy()
        gpoe_lower = (gpoe_m - 2 * torch.sqrt(gpoe_v)).detach().numpy()  # - 2*model_2.likelihood.sigma.detach().numpy()

        plt.plot(x_test, gpoe, 'm-', linewidth=1.5)
        plt.plot(x_test, gpoe_upper, 'm-', linewidth=2.5)
        plt.plot(x_test, gpoe_lower, 'm-', linewidth=2.5)

        bcm = bcm_m.detach().numpy()
        bcm_upper = (bcm_m + 2 * torch.sqrt(bcm_v)).detach().numpy()  # + 2*model_2.likelihood.sigma.detach().numpy()
        bcm_lower = (bcm_m - 2 * torch.sqrt(bcm_v)).detach().numpy()  # - 2*model_2.likelihood.sigma.detach().numpy()

        plt.plot(x_test, bcm, 'r-', linewidth=1.5)
        plt.plot(x_test, bcm_upper, 'r-', linewidth=2.5)
        plt.plot(x_test, bcm_lower, 'r-', linewidth=2.5)

        rbcm = rbcm_m.detach().numpy()
        rbcm_upper = (rbcm_m + 2 * torch.sqrt(rbcm_v)).detach().numpy()  # + 2*model_2.likelihood.sigma.detach().numpy()
        rbcm_lower = (rbcm_m - 2 * torch.sqrt(rbcm_v)).detach().numpy()  # - 2*model_2.likelihood.sigma.detach().numpy()

        plt.plot(x_test, rbcm, 'b-', linewidth=1.5)
        plt.plot(x_test, rbcm_upper, 'b-', linewidth=2.5)
        plt.plot(x_test, rbcm_lower, 'b-', linewidth=2.5)

        plt.title(r'Variational Sparse GP -- (task=' + str(k + 1) + ')')
        plt.xlabel(r'Input, $x$')
        plt.ylabel(r'Output, $y$')
        plt.xlim(min_x - 0.5, max_x + 0.5)
        plt.ylim(-22.0, 22.0)

        if save:
            plt.savefig(fname='./figs/baseline/million_ensemble.pdf',format='pdf')

        plt.show()


