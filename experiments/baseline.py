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
#
# -----------------------------------------------------------------
# Experiment -- Baselines
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

#experiment = '10k'
experiment = '100k'
#experiment = '1m'

if experiment == '10k':
    node_overlapping = 1
    N_k = 200
    trials = 10
elif experiment == '100k':
    node_overlapping = 5
    N_k = 400
    trials = 10
elif experiment == '1m':
    node_overlapping = 100
    N_k = 800
    trials = 10
else:
    raise ValueError('Experiment indicator not valid! Must be {10k, 100k or 1m}')

M_k = 3
M_e = 35
plot_local = True
plot_ensemble = True
save = True

recy_metrics = np.zeros((3,trials))
poe_metrics = np.zeros((3,trials))
gpoe_metrics = np.zeros((3,trials))
bcm_metrics = np.zeros((3,trials))
rbcm_metrics = np.zeros((3,trials))

for trial in range(trials):

    tasks = 50
    T = 50

    print('TRIAL = '+str(trial)+'/'+str(trials))

    ###########################
    #                         #
    #    DISTRIBUTED TASKS    #
    #                         #
    ###########################

    min_x = 0.0
    max_x = T * 0.1
    segment_x = (max_x - min_x) / tasks
    x_tasks = []
    y_tasks = []
    for n in range(node_overlapping):
        for k in range(T):
            x_k = ((min_x + (k * segment_x)) - (min_x + ((k + 1) * segment_x))) * torch.rand(N_k, 1) + (
                        min_x + ((k + 1) * segment_x))
            x_k, _ = torch.sort(x_k, dim=0)
            y_k = smooth_function(x_k) + 2.0 * torch.randn(N_k, 1)
            x_tasks.append(x_k)
            y_tasks.append(y_k)

    tasks = T * node_overlapping

    print('# of tasks: ', tasks)

    ###########################
    #                         #
    #   PARALLEL INFERENCE    #
    #                         #
    ###########################

    N_k_test = 400
    x_test = torch.linspace(min_x-0.5, max_x+0.5, N_k_test)[:, None]
    models = []       # for recyclable GPs
    models_dist = []  # for distributed GPs
    x_all = []        # for distributed GPs
    y_all = []        # for distributed GPs
    for k, x_k in enumerate(x_tasks):
        print('-                             -')
        print('----- TASK k='+str(k+1)+' ------')
        print('-                             -')
        ######################################################
        # 1. RECYCLABLE GP
        ######################################################
        kernel_k = RBF()
        likelihood_k = Gaussian(fit_noise=False)
        model_k = SVGP(kernel_k, likelihood_k, M_k)

        z_k_min = min_x + ((k%T)*segment_x)
        z_k_max = min_x + (((k%T)+1)*segment_x)
        model_k.z = torch.nn.Parameter(torch.linspace(z_k_min, z_k_max, M_k)[:, None], requires_grad=True)
        vem_algorithm = AlgorithmVEM(model_k, x_k, y_tasks[k], iters=15)

        vem_algorithm.ve_its = 20
        vem_algorithm.vm_its = 10
        vem_algorithm.lr_m = 1e-6
        vem_algorithm.lr_L = 1e-10
        vem_algorithm.lr_hyp = 1e-10
        vem_algorithm.lr_z = 1e-10

        vem_algorithm.fit()
        models.append(model_k)

        ######################################################
        # 2. DISTRIBUTED GP (FOR BCM, RBCM, POE & GPOE)
        ######################################################

        kernel_j = RBF()
        likelihood_j = Gaussian(fit_noise=True)
        model_j = DistGP(kernel_j, likelihood_j)
        GPR_Optimizer(model_j, x_k, y_tasks[k])

        models_dist.append(model_j)
        x_all.append(x_k)
        y_all.append(y_tasks[k])

        if plot_local:
            gp, gp_upper, gp_lower = model_k.predictive(x_test)
            disgp_m, disgp_v = model_j.predictive(x_k, y_tasks[k], x_test)

            disgp = disgp_m.detach().numpy()
            disgp_upper = (disgp_m + 2 * torch.sqrt(disgp_v)).detach().numpy() + 2 * model_j.likelihood.sigma.detach().numpy()
            disgp_lower = (disgp_m - 2 * torch.sqrt(disgp_v)).detach().numpy() - 2 * model_j.likelihood.sigma.detach().numpy()

            plt.figure(figsize=(12, 4))
            plt.plot(x_k, y_tasks[k], ls='-', color=color_palette[k%len(color_palette)], markersize=2.5, markeredgewidth=0.75)
            plt.plot(models[k].z.detach(), -20.0*torch.ones(M_k, 1), color=color_palette[k%len(color_palette)], linestyle='', marker='.',markersize=5)

            plt.plot(x_test, gp, 'k-', linewidth=1.5)
            #plt.fill_between(x_test.flatten(), gp_lower.flatten(), gp_upper.flatten(), color='b', alpha=0.2,lw='0.5')
            plt.plot(x_test, gp_upper, 'k-', linewidth=2.5)
            plt.plot(x_test, gp_lower, 'k-', linewidth=2.5)

            plt.plot(x_test, disgp, 'b-', linewidth=1.5)
            #plt.fill_between(x_test.flatten(), gp_lower.flatten(), gp_upper.flatten(), color='b', alpha=0.2,lw='0.5')
            plt.plot(x_test, disgp_upper, 'b-', linewidth=2.5)
            plt.plot(x_test, disgp_lower, 'b-', linewidth=2.5)

            plt.title(r'Variational Sparse GP -- (task=' + str(k+1) + ')')
            plt.xlabel(r'Input, $x$')
            plt.ylabel(r'Output, $y$')
            plt.xlim(min_x - 0.5, max_x + 0.5)
            plt.ylim(-22.0, 22.0)

            if save:
                plt.savefig(fname='./figs/baseline/distributed_task_'+str(k+1)+'.pdf',format='pdf')

            plt.close()
            #plt.show()

    ###########################
    #                         #
    #   ENSEMBLE INFERENCE    #
    #                         #
    ###########################
    print('-                   -')
    print('----- ENSEMBLE ------')
    print('-                   -')

    # TEST DATA FOR EVALUATION
    N_e_test = 400
    x_test_ensemble = torch.linspace(min_x-0.5, max_x+0.5, N_e_test)[:, None]
    f_test_ensemble = smooth_function(x_test_ensemble)
    y_test_ensemble = f_test_ensemble + 2.0*torch.randn(N_e_test,1)

    ######################################################
    # 1. RECYCLABLE GP
    ######################################################

    kernel = RBF()
    likelihood = Gaussian(fit_noise=False)
    model_e = EnsembleGP(kernel, likelihood, models, M_e)
    model_e.z = torch.nn.Parameter(torch.linspace(min_x, max_x, M_e)[:,None], requires_grad=True)
    vem_algorithm = AlgorithmVEM(model_e, config='ensemble', iters=10)

    vem_algorithm.ve_its = 30
    vem_algorithm.vm_its = 10
    vem_algorithm.lr_m = 1e-3
    vem_algorithm.lr_L = 1e-6
    vem_algorithm.lr_hyp = 1e-8
    vem_algorithm.lr_z = 1e-8

    vem_algorithm.fit()

    nlpd = model_e.nlpd(x_test_ensemble, y_test_ensemble)
    rmse = model_e.rmse(x_test_ensemble, f_test_ensemble)
    mae = model_e.mae(x_test_ensemble, f_test_ensemble)

    recy_metrics[0, trial] = nlpd
    recy_metrics[1, trial] = rmse
    recy_metrics[2, trial] = mae

    print('Recyclable - NLPD: ', nlpd)
    print('Recyclable - RMSE: ', rmse)
    print('Recyclable - MAE: ',  mae)
    print(' ')

    ######################################################
    # 2. DISTRIBUTED GP (FOR BCM, RBCM, POE & GPOE)
    ######################################################

    # A. POE  _________//

    poe_model = PoeGP(models_dist)

    nlpd = poe_model.nlpd(x_all, y_all, x_test_ensemble, y_test_ensemble)
    rmse = poe_model.rmse(x_all, y_all, x_test_ensemble, f_test_ensemble)
    mae = poe_model.mae(x_all, y_all, x_test_ensemble, f_test_ensemble)

    poe_metrics[0, trial] = nlpd
    poe_metrics[1, trial] = rmse
    poe_metrics[2, trial] = mae

    print('POE-NLPD: ', nlpd)
    print('POE-RMSE: ', rmse)
    print('POE-MAE: ',  mae)
    print(' ')

    # B. GPOE _________//

    gpoe_model = GenPoeGP(models_dist)

    nlpd = gpoe_model.nlpd(x_all, y_all, x_test_ensemble, y_test_ensemble)
    rmse = gpoe_model.rmse(x_all, y_all, x_test_ensemble, f_test_ensemble)
    mae = gpoe_model.mae(x_all, y_all, x_test_ensemble, f_test_ensemble)

    gpoe_metrics[0, trial] = nlpd
    gpoe_metrics[1, trial] = rmse
    gpoe_metrics[2, trial] = mae

    print('GenPOE-NLPD: ', nlpd)
    print('GenPOE-RMSE: ', rmse)
    print('GenPOE-MAE: ',  mae)
    print(' ')

    # C. BCM  _________//

    bcm_model = BayesianCM(models_dist)

    nlpd = bcm_model.nlpd(x_all, y_all, x_test_ensemble, y_test_ensemble)
    rmse = bcm_model.rmse(x_all, y_all, x_test_ensemble, f_test_ensemble)
    mae = bcm_model.mae(x_all, y_all, x_test_ensemble, f_test_ensemble)

    bcm_metrics[0, trial] = nlpd
    bcm_metrics[1, trial] = rmse
    bcm_metrics[2, trial] = mae

    print('BCM-NLPD: ', nlpd)
    print('BCM-RMSE: ', rmse)
    print('BCM-MAE: ',  mae)
    print(' ')

    # D. RBCM _________//

    rbcm_model = RobustBayesianCM(models_dist)

    nlpd = rbcm_model.nlpd(x_all, y_all, x_test_ensemble, y_test_ensemble)
    rmse = rbcm_model.rmse(x_all, y_all, x_test_ensemble, f_test_ensemble)
    mae = rbcm_model.mae(x_all, y_all, x_test_ensemble, f_test_ensemble)

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
        gp, gp_upper, gp_lower = model_e.predictive(x_test_ensemble)

        poe_m, poe_v = poe_model.predictive(x_all, y_all, x_test_ensemble)
        gpoe_m, gpoe_v = gpoe_model.predictive(x_all, y_all, x_test_ensemble)
        bcm_m, bcm_v = gpoe_model.predictive(x_all, y_all, x_test_ensemble)
        rbcm_m, rbcm_v = rbcm_model.predictive(x_all, y_all, x_test_ensemble)

        # Plot Ensemble
        plt.figure(figsize=(12, 4))
        for k in range(50):
            #if k%10==0:
            plt.plot(x_tasks[k], y_tasks[k], ls='-', color=color_palette[k%len(color_palette)], markersize=2.5, markeredgewidth=0.75)
            plt.plot(models[k].z.detach(), -20.0*torch.ones(M_k,1), color=color_palette[k%len(color_palette)], linestyle='', marker='.', markersize=5)

        plt.plot(model_e.z.detach(), -20.0 * torch.ones(M_e, 1), color='r', linestyle='', marker='x', markersize=5, markeredgewidth=1.0)
        plt.plot(x_test_ensemble, gp, 'k-', linewidth=1.5)
        plt.plot(x_test_ensemble, gp_upper, 'k-', linewidth=2.5)
        plt.plot(x_test_ensemble, gp_lower, 'k-', linewidth=2.5)

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

        plt.title(r'Ensemble GP Model -- (tasks='+str(tasks)+')')
        plt.xlabel(r'Input, $x$')
        plt.ylabel(r'Output, $y$')
        plt.xlim(min_x-0.5, max_x+0.5)
        plt.ylim(-22.0, 22.0)

        if save:
            plt.savefig(fname='./figs/baseline/distributed_ensemble.pdf',format='pdf')

        #plt.show()
        plt.close()