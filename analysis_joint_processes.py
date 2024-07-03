from simulator import ParticleSimulator, SimulationParameters
import numpy as np
from scipy import stats
from scipy.linalg import expm
import os, time
import matplotlib.pyplot as plt

from diffusion import get_ddpm_params, q_sample
from default_config import config as config

np.random.seed(123)


num_procs = 10000

z0 = np.zeros((num_procs, 2))


zs = []
## Simulations
params = SimulationParameters(
    num_procs=num_procs,
    sigma_noise=5.,
    tau_x=4,
    tau_y=5,
    C=50,
)

simulator = ParticleSimulator(z0, params)
simulator.simulate(2000)

zs = simulator.z
zs = zs[:30000:10][-1024:].transpose(1,0,2)

ddpm_params = get_ddpm_params(config.ddpm)
ddpm_params = {k: np.array(v) for k, v in ddpm_params.items()}

##

def get_noised_process(z, t, ddpm_params):
    noise = np.random.randn(*zs.shape)
    return q_sample(z, t, noise, ddpm_params)

z_t = get_noised_process(zs, 0, ddpm_params)

## 

def compute_covariance(t, delta_s, ddpm_params):
    A = np.array([[1 / params.tau_y, 0               ],
                  [-1 / params.C,         1 / params.tau_x]])
    s1 = params.sigma2_noise / (2 * A[0,0]) 
    s2 = -( A[1,0] * params.sigma2_noise ) / ( 2 * A[0,0] * (A[0,0] + A[1,1]) )
    s3 = ( A[1,0]**2 * params.sigma2_noise ) / ( 2 * A[0,0] * A[1,1] * (A[0,0] + A[1,1]) )
    Sigma = np.array([[s1, s2],
                      [s2, s3]])

    expAdelta = expm(-A * delta_s)

    return Sigma.dot(expAdelta.T) * ddpm_params["alphas_bar"][t] + (1 - ddpm_params["alphas_bar"][t]) * np.eye(2)

## 

def compute_crosscov(x1, x2):
    return (x1[:,:,None] * x2[:,None,:]).mean(0)

crosscovs_theory = []
crosscovs_sim = []
for delta in range(20):
    x1 = z_t[:,500,:]
    x2 = z_t[:,500+delta,:]

    crosscovs_theory.append(compute_covariance(0, delta, ddpm_params))
    crosscovs_sim.append(compute_crosscov(x1, x2))
crosscovs_theory = np.array(crosscovs_theory)    
crosscovs_sim = np.array(crosscovs_sim)    

plt.plot(crosscovs_theory.reshape((20, -1)))
plt.plot(crosscovs_sim.reshape((20, -1)), "*")
plt.show()


