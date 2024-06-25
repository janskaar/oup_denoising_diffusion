import ml_collections


## Config for diffusion model
config = ml_collections.ConfigDict()

# training
config.training = training = ml_collections.ConfigDict()
training.num_train_steps = 5000

# ddpm
config.ddpm = ddpm = ml_collections.ConfigDict()
ddpm.beta_schedule = "linear"
ddpm.timesteps = 1000

# data
config.data = data = ml_collections.ConfigDict()
data.batch_size = 64
data.length = 1024
data.channels = 2

# model
config.model = model = ml_collections.ConfigDict()
model.use_encoder = True
model.start_filters = 16
model.filter_mults = (1, 2, 4, 8)
model.encoder_start_filters = 16
model.encoder_filter_mults = (1, 2, 4, 8)
model.encoder_latent_dim = 4
model.use_attention = False

# optim
config.optim = optim = ml_collections.ConfigDict()
optim.optimizer = "Adam"
optim.learning_rate = 1e-4
optim.beta1 = 0.9
optim.beta2 = 0.999
optim.eps = 1e-8
optim.warmup_steps = 500
optim.use_full_loss = False

config.seed = 123
