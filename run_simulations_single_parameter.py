from simulator import ParticleSimulator, SimulationParameters
import numpy as np
from scipy import stats
import os, time

np.random.seed(123)

prior_min = np.array([1, 1, 1, -250])
prior_max = np.array([10, 10, 10, 250])

num_sims = 10000

param_vals = np.tile(np.array([[2., 15., 2., 5.]]), (num_sims, 1))


z0 = np.zeros((1, 2))
zs = []
## Simulations
for i, _ in enumerate(param_vals):
    print(i, end="\r")
    params = SimulationParameters(
        num_procs=1,
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

np.save(os.path.join("data", "theta_single_param.npy"), param_vals)
np.save(os.path.join("data", "z_single_param.npy"), zs)

