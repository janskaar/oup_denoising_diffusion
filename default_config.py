import ml_collections
import os


## Config for diffusion model
config = ml_collections.ConfigDict()

# training
config.training = training = ml_collections.ConfigDict()
training.num_train_steps = 5000
training.eval_every = 1000
training.num_warmup_steps = 500
training.use_full_loss = False


# ddpm
config.ddpm = ddpm = ml_collections.ConfigDict()
ddpm.beta_schedule = "linear"
ddpm.timesteps = 1000

# data
config.data = data = ml_collections.ConfigDict()
data.batch_size = 64
data.length = 1024
data.channels = 2
data.X_train_path = os.path.join("data", "z.npy")
data.Theta_train_path = os.path.join("data", "theta.npy")
data.X_fixed_points_path = os.path.join("data", "z_fixed_points.npy")
data.Theta_fixed_points_path = os.path.join("data", "theta_fixed_points.npy")
data.norm_axis = 1 # data has shape (N, time, channel)


# model
config.model = model = ml_collections.ConfigDict()
model.use_encoder = True
model.use_parameters = False
model.start_filters = 16
model.filter_mults = (1, 2, 4, 8)
model.encoder_start_filters = 16
model.encoder_filter_mults = (1, 2, 4, 8)
model.encoder_latent_dim = 4

# optim
config.optim = optim = ml_collections.ConfigDict()
optim.optimizer = "Adam"
optim.learning_rate = 1e-4
optim.beta1 = 0.9
optim.beta2 = 0.999
optim.eps = 1e-8


config.seed = 123
