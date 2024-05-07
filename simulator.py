import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass


@dataclass
class SimulationParameters:
    dt: float = 0.1
    num_procs: int = 1000000

    sigma_noise: float = 1.0
    tau_x: float = 20.0
    tau_y: float = 4.0
    C: float = 250.0

    @property
    def sigma2_noise(self):
        return self.sigma_noise ** 2

    @property
    def tau_tilde(self):
        return 1 / (1 / self.tau_x + 1 / self.tau_y)

    @property
    def R(self):
        return self.tau_x / self.C


class ParticleSimulator:
    def __init__(self, z_0, params):
        self.p = params
        self.z_0 = z_0
        self._step = 0
        self.compute_propagators()


    def compute_propagators(self):
        A = np.array(
            [[-1.0 / self.p.tau_y, 0.0], [1.0 / self.p.C, -1.0 / self.p.tau_x]]
        )
        self.expectation_prop = expm(A * self.p.dt)


    def simulate(self, t):
        num_steps = int(t / self.p.dt)
        self.num_steps = num_steps

        self.z = np.zeros((num_steps + 1, self.p.num_procs, 2), dtype=np.float64)
        self.z[0] = self.z_0

        self._step += 1
        for _ in range(num_steps):
            i = self._step

            self.propagate()
            self._step += 1

    def propagate(self):
        i = self._step
        self.z[i] = self.expectation_prop.dot(self.z[i - 1].T).T
        self.z[i, :, 0] += (
            np.random.randn(self.p.num_procs) * self.p.sigma_noise * np.sqrt(self.p.dt)
        )

