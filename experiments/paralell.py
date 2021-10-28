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
# Experiment -- Parallel Inference
# -----------------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
from tikzplotlib import save as tikz_save

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
from optimization.algorithms import vem_algorithm, ensemble_vem, ensemble_vem_parallel
from optimization.algorithms import AlgorithmVEM
from util import smooth_function, smooth_function_bias

tasks = 5
N_k = 500
M_k = 15
M_e = 35
plot_local = True
plot_ensemble = True
save = True

###########################
#                         #
#    DISTRIBUTED TASKS    #
#                         #
###########################

min_x = 0.0
max_x = 5.5
segment_x = (max_x - min_x)/tasks
x_tasks = []
y_tasks = []
for k in range(tasks):
    x_k = ((min_x+(k*segment_x))-(min_x+((k+1)*segment_x)))*torch.rand(N_k,1) + (min_x+((k+1)*segment_x))
    x_k, _ = torch.sort(x_k, dim=0)
    y_k = smooth_function_bias(x_k) + 2.0*torch.randn(N_k,1)
    x_tasks.append(x_k)
    y_tasks.append(y_k)

###########################
#                         #
#   PARALLEL INFERENCE    #
#                         #
###########################

N_k_test = 400
x_test = torch.linspace(min_x-0.5, max_x+0.5, N_k_test)[:, None]
models = []
for k, x_k in enumerate(x_tasks):
    print('-                             -')
    print('----- TASK k='+str(k+1)+' ------')
    print('-                             -')
    kernel_k = RBF()
    likelihood_k = Gaussian(fit_noise=False)
    model_k = SVGP(kernel_k, likelihood_k, M_k)

    z_k_min = min_x+(k*segment_x)
    z_k_max = min_x+((k+1)*segment_x)
    #model_k.z = torch.nn.Parameter((z_k_max - z_k_min)*torch.rand(M_k, 1) + z_k_min, requires_grad=True)
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

    if plot_local:
        gp, gp_upper, gp_lower = model_k.predictive(x_test)

        plt.figure(figsize=(12, 4))
        plt.plot(x_k, y_tasks[k], ls='-', color=color_palette[k], lw=1.5)
        plt.plot(models[k].z.detach(), -20.0*torch.ones(M_k, 1), color=color_palette[k], linestyle='', marker='.',markersize=5)

        plt.plot(x_test, gp, 'k-', linewidth=1.5)
        #plt.fill_between(x_test.flatten(), gp_lower.flatten(), gp_upper.flatten(), color='b', alpha=0.2,lw='0.5')
        plt.plot(x_test, gp_upper, 'k-', linewidth=3.0)
        plt.plot(x_test, gp_lower, 'k-', linewidth=3.0)

        plt.title(r'Variational Sparse GP -- (task=' + str(k+1) + ')')
        plt.xlabel(r'Input, $x$')
        plt.ylabel(r'Output, $y$')
        plt.xlim(min_x - 0.5, max_x + 0.5)
        plt.ylim(-22.0, 22.0)

        if save:
            plt.savefig(fname='./figs/ parallel_task_'+str(k+1)+'.pdf',format='pdf')

        plt.show()

###########################
#                         #
#   ENSEMBLE INFERENCE    #
#                         #
###########################
print('-                   -')
print('----- ENSEMBLE ------')
print('-                   -')

kernel = RBF()
likelihood = Gaussian(fit_noise=False)
model_e = EnsembleGP(kernel, likelihood, models, M_e)
model_e.z = torch.nn.Parameter(torch.linspace(min_x, max_x, M_e)[:,None], requires_grad=True)
vem_algorithm = AlgorithmVEM(model_e, config='ensemble', iters=30)

vem_algorithm.ve_its = 30
vem_algorithm.vm_its = 10
vem_algorithm.lr_m = 1e-3
vem_algorithm.lr_L = 1e-6
vem_algorithm.lr_hyp = 1e-8
vem_algorithm.lr_z = 1e-8

vem_algorithm.fit()

N_e_test = 400
x_test_ensemble = torch.linspace(min_x-0.5, max_x+0.5, N_e_test)[:, None]

if plot_ensemble:
    gp, gp_upper, gp_lower = model_e.predictive(x_test_ensemble)

    # Plot Ensemble
    plt.figure(figsize=(12, 4))
    for k, x_k in enumerate(x_tasks):
        #if k%10==0:
        plt.plot(x_k, y_tasks[k], ls='-', color=color_palette[k], lw=1.5)
        plt.plot(models[k].z.detach(), -20.0*torch.ones(M_k,1), color=color_palette[k], linestyle='', marker='.', markersize=5)

    plt.plot(model_e.z.detach(), -20.0 * torch.ones(M_e, 1), color='k', linestyle='', marker='x', markersize=7, markeredgewidth=1.1)
    plt.plot(x_test_ensemble, gp, 'k-', linewidth=1.5)
    #plt.fill_between(x_test_ensemble.flatten(), gp_lower.flatten(), gp_upper.flatten(), color='b', alpha=0.2, lw='0.5')
    plt.plot(x_test_ensemble, gp_upper, 'k-', linewidth=3.0)
    plt.plot(x_test_ensemble, gp_lower, 'k-', linewidth=3.0)

    plt.title(r'Ensemble GP Model -- (tasks='+str(tasks)+')')
    plt.xlabel(r'Input, $x$')
    plt.ylabel(r'Output, $y$')
    plt.xlim(min_x-0.5, max_x+0.5)
    plt.ylim(-22.0, 22.0)

    if save:
        plt.savefig(fname='./figs/parallel_ensemble.pdf',format='pdf')

    plt.show()

    N_e_test = 400
    x_test_ensemble = torch.linspace(min_x-0.5, max_x+0.5, N_e_test)[:, None]
    f_test_ensemble = smooth_function(x_test_ensemble)
    y_test_ensemble = f_test_ensemble + 2.0*torch.randn(N_e_test,1)

    nlpd = model_e.nlpd(x_test_ensemble, y_test_ensemble)
    rmse = model_e.rmse(x_test_ensemble, f_test_ensemble)
    mae = model_e.mae(x_test_ensemble, f_test_ensemble)

    print("NLPD: ", nlpd)
    print("RMSE: ", rmse)
    print("MAE: ", mae)