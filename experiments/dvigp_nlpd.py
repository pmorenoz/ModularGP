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
# Experiment -- Baselines / Y. Gal et al. (2014)
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
from baselines.distgp import DistGP
from baselines.dvigp import DVIGP
from optimization.algorithms import GPR_Optimizer
#from models.svgp import predictive
from optimization.algorithms import vem_algorithm
from util import smooth_function

experiment = '10k'
#experiment = '100k'
#experiment = '1m'

if experiment == '10k':
    node_overlapping = 1
    N_k = 200
    trials = 10
    N = 10000
elif experiment == '100k':
    node_overlapping = 5
    N_k = 400
    trials = 10
    N = 100000
elif experiment == '1m':
    node_overlapping = 100
    N_k = 800
    trials = 10
    N = 1000000
else:
    raise ValueError('Experiment indicator not valid! Must be {10k, 100k or 1m}')

M = 35
plot_local = False
plot_ensemble = False
save = False

dvigp_metrics = np.zeros((3,trials))

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
    x = (min_x - max_x)*torch.rand(N, 1) + max_x
    x, _ = torch.sort(x, dim=0)
    y = smooth_function(x) + 2.0*torch.randn(N, 1)

    tasks = T * node_overlapping

    print('Number # of tasks: ', tasks)

    ######################################################
    # 1. DISTRIBUTED VIGP (Gal 2014)
    ######################################################

    kernel_j = RBF()
    likelihood_j = Gaussian(fit_noise=True)

    model = DVIGP(kernel_j, likelihood_j, M, nodes=tasks)
    model.z = torch.nn.Parameter(torch.linspace(min_x, max_x, M)[:,None], requires_grad=True)
    vem_algorithm(model, x, y, em_iters=20, plot=False)

    # TEST DATA FOR EVALUATION
    N_e_test = 400
    x_test_ensemble = torch.linspace(min_x-0.5, max_x+0.5, N_e_test)[:, None]
    f_test_ensemble = smooth_function(x_test_ensemble)
    y_test_ensemble = f_test_ensemble + 2.0*torch.randn(N_e_test,1)

    nlpd = model.nlpd(x_test_ensemble, y_test_ensemble)
    rmse = model.rmse(x_test_ensemble, f_test_ensemble)
    mae = model.mae(x_test_ensemble, f_test_ensemble)

    dvigp_metrics[0, trial] = nlpd
    dvigp_metrics[1, trial] = rmse
    dvigp_metrics[2, trial] = mae

    print('Distributed VIGP - NLPD: ', nlpd)
    print('Distributed VIGP - RMSE: ', rmse)
    print('Distributed VIGP - MAE: ',  mae)
    print(' ')


