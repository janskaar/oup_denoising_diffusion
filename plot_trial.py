import h5py, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch

top_dir = os.path.join("results_simple_loss_32_16")
files = sorted([os.path.join(top_dir, f) for f in os.listdir(top_dir) if ".h5" in f])
num_runs = len(files)

## 

lrs = []
losses = []
samples = []

for j, f in enumerate(files):
    with h5py.File(f, "r") as f:
        keys = list(f.keys())
        print("use_full_loss: ", f["config"].attrs["training.use_full_loss"])
        steps = [key for key in keys if key.isnumeric()] 
        step_losses = []
        step_samples = []
        for i, step in enumerate(steps):
            step_losses.append(f[step]["full_chain_loss"][()])
            if i > 0:
                step_samples.append(f[step]["samples"][()])
        lrs.append(f["config"].attrs["optim.learning_rate"])
        losses.append(step_losses)
        samples.append(step_samples)
        if j == 0:
            conditions = f["conditions"][()]

losses = np.array(losses)
samples = np.array(samples)
lrs = np.array(lrs)
steps = np.array([int(k) for k in keys if k.isnumeric()])

## Plot mean (over all axes) losses for each run

fig, ax = plt.subplots(1)
for loss in losses:
    plt.plot(loss.mean((1,2)), "*")

ax.set_ylim(0, 2)
plt.show()

## Plot mean (over samples) losses for each run and each step

n = len(steps)
fig, ax = plt.subplots(n, sharex=True, sharey=False)
for i in range(n):
    for loss in losses:
        ax[i].plot(loss[i].mean(1))
    if i == 0:
        ax[i].set_ylim(0, 8)
    else:
        ax[i].set_ylim(0, 1.2)

plt.show()

## Plot mean PSD for samples

phi = np.arctan(1080 / 1920)
diag = 20
sz = (diag * np.cos(phi), diag * np.sin(phi))


run_ids = [f.split("/")[1][:-3] for f in files]
fs, psd_cond = welch(conditions.transpose(0,2,1).reshape((-1, 50, 2, 1024)), fs=1000)
for i in range(num_runs):
    fig = plt.figure()
    outer_grid = gridspec.GridSpec(3, 3, left=0.03, right=0.99, bottom=0.05, top=0.98)
    fig.set_size_inches(*sz)

    # Plot PSDs
    _, psd_sample = welch(samples[i,-1].transpose(0,2,1).reshape((-1,50,2,1024)), fs=1000)
    psd_grid = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer_grid[:2,:2])
    for j in range(9):
        sharex = None if j == 0 else ax
        sharey = None if j == 0 else ax
        ax = fig.add_subplot(psd_grid[j // 3, j % 3], sharex=sharex, sharey=sharey)
        ax.plot(fs, psd_cond[j,:,0,:].mean(0))
        ax.plot(fs, psd_sample[j,:,0,:].mean(0))
        if j >= 6:
            ax.set_xlabel("Frequency [Hz]")


    # Plot mean losses for all runs
    ax = fig.add_subplot(outer_grid[:2,2])
    mean_losses = losses.mean((2,3)) # shape (num_runs, num_saved_steps)
    ymin = mean_losses[:,-1].min() * 0.99
    ymax = mean_losses[mean_losses[:,2] < 1, 2].max()
    for j in range(num_runs):
        lw = 4 if j == i else None
        color = "black" if j == i else None
        ax.semilogy(steps, mean_losses[j], lw=lw, color=color, label=run_ids[j])
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean loss")
    ax.set_ylim(ymin, ymax)
    ax.legend()


    fig.savefig(os.path.join(top_dir, "sample_psds_" + run_ids[i] + ".svg"))
#    plt.show()
    plt.close()
#    break








