from simulator import ParticleSimulator, SimulationParameters
import numpy as np
from scipy import stats
import os, time

prior_min = np.array([1, 1, 1, -250])
prior_max = np.array([10, 10, 10, 250])
scale = prior_max - prior_min
prior = stats.uniform(loc=prior_min, scale=scale)

num_procs = 1



z0 = np.zeros((num_procs, 2))

num_sims = 100

zs = []
## Simulations
for i in range(num_sims):
    print(i, end="\r")
    param_vals = prior.rvs()
    params = SimulationParameters(
            num_procs = num_procs,
            sigma_noise = param_vals[0],
            tau_x = param_vals[1],
            tau_y = param_vals[2],
            C = param_vals[3]
    )

    simulator = ParticleSimulator(z0, params)
    simulator.simulate(500)
    zs.append(simulator.z.squeeze()) 
zs = np.array(zs)

