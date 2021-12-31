import os
import torch

# Root directory for dataset
dataroot = "/content/drive/Shareddrives/DL training/IMVFX_final/GID"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 1
cut_batch_size = 16

# The size of images 
image_size = 256

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 400

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# The weight for PatchNCE loss and identity loss
lamb_x = 1
lamb_y = 1

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Save checkpoints every few epochs
save_steps = 4

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
