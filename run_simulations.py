from simulator import ParticleSimulator, SimulationParameters
import numpy as np
from scipy import stats
import os, time

np.random.seed(123)

prior_min = np.array([1, 1, 1, -250])
prior_max = np.array([10, 10, 10, 250])
scale = prior_max - prior_min
prior = stats.uniform(loc=prior_min, scale=scale)

num_procs = 1

z0 = np.zeros((num_procs, 2))

num_sims = 2000
param_vals = prior.rvs(size=(num_sims, 4))

# # #  For running multiple simulations at same parameter values
# param_vals = prior.rvs(size=(9, 4))
# param_vals = np.tile(param_vals, (50, 1))
# param_vals = param_vals.reshape((50, 9, 4)).transpose(1,0,2).reshape((450,4))

zs = []
## Simulations
for i, _ in enumerate(param_vals):
    print(i, end="\r")
    params = SimulationParameters(
        num_procs=num_procs,
        sigma_noise=param_vals[i,0],
        tau_x=param_vals[i,1],
        tau_y=param_vals[i,2],
        C=param_vals[i,3],
    )

    simulator = ParticleSimulator(z0, params)
    simulator.simulate(3000)
    zs.append(simulator.z.squeeze())

zs = np.array(zs)

zs = zs[:, :30000:10, :]

np.save(os.path.join("data", "theta_test_set.npy"), param_vals)
np.save(os.path.join("data", "z_test_set.npy"), zs)
