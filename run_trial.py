from training_loop import train
import numpy as np
from scipy.stats.qmc import Halton
import os, time
from pathlib import Path
from default_config import config as default_config

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
job_id = int(os.environ["SLURM_ARRAY_JOB_ID"])
num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
base_seed = job_id + task_id


path = Path(__file__).parent
print("base seed: ", base_seed, flush=True)


np.random.seed(base_seed)

batch_size = 128
default_config.training.use_full_loss = False
default_config.model.start_filters = 32
default_config.model.encoder_start_filters = 16
default_config.model.encoder_latent_dim = 4
default_config.model.use_encoder = True
default_config.model.use_parameters = False
default_config.model.normalization = True

default_config.data.batch_size = batch_size
default_config.data.norm_scale = 1.743

default_config.training.num_train_steps = 5000
default_config.training.num_warmup_steps = 500

default_config.data.batch_size = batch_size

default_config.data.prior_min = (1, 5, 1, 5) # (sigma2_noise, tau_x, tau_y, c)
default_config.data.prior_max = (5, 10, 5, 10)



outdir = os.path.join(path, "results_latent_4_global_norm_small_parameterspace")

runs_per_task = 20

if task_id == 0:
    # Generate adam parameters to be marginalized
    sampler = Halton(d=2, scramble=True, seed=np.random.randint(2 ** 32))
    sample = sampler.random(n=runs_per_task * num_tasks)
    learning_rates = sample[:,0]
    beta1s = sample[:,1]

    # log space
    learning_rates *= (np.log(1e-1) - np.log(1e-5))
    learning_rates += np.log(1e-5)
    learning_rates = np.exp(learning_rates).squeeze()
    
    # linear space
    beta1s *= (0.98 - 0.5)
    beta1s += 0.5
    beta1s = beta1s.squeeze()

    np.save(os.path.join(outdir, "learning_rates.npy"), learning_rates)
    np.save(os.path.join(outdir, "beta1s.npy"), beta1s)
else:
    # Load adam parameters to be marginalized
    for i in range(100):
        try:
            learning_rates = np.load(os.path.join(outdir, "learning_rates.npy"))
            beta1s = np.load(os.path.join(outdir, "beta1s.npy"))
        except FileNotFoundError:
            time.sleep(1.)

learning_rates = learning_rates[task_id * runs_per_task:(task_id+1) * runs_per_task]
beta1s = beta1s[task_id * runs_per_task:(task_id+1) * runs_per_task]

print("")
print("====================", flush=True)
print(f"Task {task_id}", flush=True)
print(f"Learning rates:", flush=True)
print(learning_rates, flush=True)
print(f"beta1s:", flush=True)
print(beta1s, flush=True)
print("====================", flush=True)


for i, lr in enumerate(learning_rates):
    config = default_config.copy_and_resolve_references()

    config.optim.learning_rate = lr
    config.optim.beta1 = beta1s[i] 

    config.seed = np.random.randint(2 ** 32)
    config.training.eval_file = os.path.join(outdir, f"run_{i + task_id * runs_per_task}.h5")
    config.training.checkpoint_dir = os.path.join(outdir, f"checkpoint_{i + task_id * runs_per_task}")
    train(config)

