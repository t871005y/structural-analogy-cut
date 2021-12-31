import os
import glob 
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep

from . import hyperparameters as hyperparams
from . import cut_model
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
def train(opt,dataloader):
    real_label = 1.0
    fake_label = 0.

    # Training Loop
    iters = 0
    G_losses = []
    D_losses = []

    cut_netG = cut_model.Generator().to(opt.device)
    cut_model.weights_init(cut_netG)
    # print(netG)

    cut_netD = cut_model.Discriminator().to(opt.device)
    cut_model.weights_init(cut_netD)
    # print(netD)

    cut_netH = cut_model.MLP().to(opt.device)
    cut_model.weights_init(cut_netH)
    # print(netH)

    cut_criterionBCE = nn.BCELoss().to(opt.device)
    cut_criterionMSE = nn.MSELoss().to(opt.device)

    cut_optimizerD = optim.Adam(cut_netD.parameters(), lr=hyperparams.lr, betas=(hyperparams.beta1, 0.999))
    cut_optimizerG = optim.Adam(cut_netG.parameters(), lr=hyperparams.lr, betas=(hyperparams.beta1, 0.999))
    cut_optimizerH = optim.Adam(cut_netH.parameters(), lr=hyperparams.lr, betas=(hyperparams.beta1, 0.999))

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(hyperparams.num_epochs):
        # progress_bar = tqdm(dataloader)
        cut_netG.train()
        cut_netD.train()
        cut_netH.train()
        # For each batch in the dataloader
        for i, data in tqdm(enumerate(dataloader),total=len(dataloader)):

            ############################
            # (1) Update D network
            ###########################
            set_requires_grad(cut_netD, True)
            ## Train with all-real batch
            cut_netD.zero_grad()
            # Format batch
            real = data[0].to(opt.device)
            g_input = data[1].to(opt.device)
            '''
            '''
            pred_real = cut_netD(real).view(-1) # view: reshape, -1: uncertion columns number
            label = torch.full_like(pred_real, real_label, dtype=torch.float, device=opt.device) # size[900] (30 * 30 patches)
            errD_real = cut_criterionMSE(pred_real, label)

            ## Train with all-fake batch
            '''
            '''
            fake = cut_netG(g_input)
            # detach: The result will never require gradient.
            pred_fake = cut_netD(fake.detach()).view(-1)
            label = torch.full_like(pred_fake, fake_label, dtype=torch.float, device=opt.device)
            errD_fake = cut_criterionMSE(pred_fake, label)

            errD = (errD_real + errD_fake) * 0.5
            errD.backward()
            cut_optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            set_requires_grad(cut_netD, False)
            cut_netG.zero_grad()
            cut_netH.zero_grad()
            '''
            '''
            # Calculate BCE loss
            fake = cut_netG(g_input)
            pred_fake = cut_netD(fake).view(-1)
            label = torch.full_like(pred_fake, real_label, dtype=torch.float, device=opt.device)
            cut_errG_GAN = cut_criterionMSE(pred_fake, label)

            # Calculate PatchNCE loss (multi-layer)
            nce_layers = [0, 4, 8, 12, 16]
            cut_errG_PatchNCE = 0.0
            fqs_enc = cut_netG(fake, enc_layer=nce_layers)
            fks_enc = cut_netG(g_input, enc_layer=nce_layers)
            for i, (featq, featk) in enumerate(zip(fqs_enc, fks_enc)):
                fq, sample_list = cut_netH(featq, enc_layer=nce_layers[i])
                fk, _ = cut_netH(featk, sample_list, enc_layer=nce_layers[i])
                cut_errG_PatchNCE += cut_model.PatchNCELoss(fq, fk, opt.device)
            cut_errG_PatchNCE /= len(nce_layers)
            
            # Calculate identity loss (multi-layer)
            cut_errG_idt = 0.0
            fake_y = cut_netG(real)
            fqs_enc = cut_netG(fake_y, enc_layer=nce_layers)
            fks_enc = cut_netG(real, enc_layer=nce_layers)
            for i, (featq, featk) in enumerate(zip(fqs_enc, fks_enc)):
                fq, sample_list = cut_netH(featq, enc_layer=nce_layers[i])
                fk, _ = cut_netH(featk, sample_list, enc_layer=nce_layers[i])
                cut_errG_idt += cut_model.PatchNCELoss(fq, fk, opt.device)
            cut_errG_idt /= len(nce_layers)

            # Sum the loss and backward
            cut_errG = cut_errG_GAN + 0.5 * (hyperparams.lamb_x * cut_errG_PatchNCE + hyperparams.lamb_y * cut_errG_idt)
            cut_errG.backward()
            cut_optimizerG.step()
            cut_optimizerH.step()
            
            # Output training stats\
            # Set the info of the progress bar
            # Note that the value of the GAN loss is not directly related to
            # the quality of the generated images.
            # progress_bar.set_infos({
            #     'Loss_D': round(errD.item(), 4),
            #     'Loss_G': round(errG.item(), 4),
            #     'Loss_G (GAN)': round(errG_GAN.item(), 4),
            #     'Loss_G (NCE_x)': round(errG_PatchNCE.item(), 4),
            #     'Loss_G (NCE_y)': round(errG_idt.item(), 4),
            #     'Epoch': epoch+1,
            #     'Step': iters,
            # })

            G_losses.append(cut_errG.item())
            D_losses.append(errD.item())
            iters += 1

        # plot_loss(G_losses, D_losses)

        # Evaluation 
        # netG.eval()
        # for idx, data in enumerate(train_dataloader):

        #     if idx % 12 == 0:

        #         input, target = data[1].to(opt.device), data[0].to(opt.device)
        #         output = netG(input)
        #         # Show result for test data
        #         fig_size = (input.size(2) * 3 / 100, input.size(3)/100)
        #         fig, axes = plt.subplots(1, 3, figsize=fig_size)
        #         imgs = [input.cpu().data, output.cpu().data, target.cpu().data]
        #         for ax, img in zip(axes.flatten(), imgs):
        #             ax.axis('off')
        #             # Scale to 0-255
        #             img = img.squeeze()
        #             img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
        #             ax.imshow(img, cmap=None, aspect='equal')
        #         axes[0].set_title("Input")
        #         axes[1].set_title("Generated")
        #         axes[2].set_title("Target")
        #         plt.subplots_adjust(wspace=0, hspace=0)
        #         fig.subplots_adjust(bottom=0)
        #         fig.subplots_adjust(top=1)
        #         fig.subplots_adjust(right=1)
        #         fig.subplots_adjust(left=0)
        #         # Save evaluation results
        #         if (epoch+1) % params.save_steps == 0:
        #             dir = os.path.join(params.log_dir, f'Epoch_{epoch+1:03d}')
        #             os.makedirs(dir, exist_ok=True)
        #             filename = os.path.join(dir, f'{idx+1}.png')
        #             plt.savefig(filename)
        #         if idx == 0:
        #             print('Show one evalutation result......')
        #             plt.show()  # Show the first result
        #         else:
        #             plt.close()
        # print('Evaluation done!')

        # Save the checkpoints.
        if (epoch+1) % hyperparams.save_steps == 0:
            netG_out_path = os.path.join(opt.out, 'netG_epoch_{}.pth'.format(epoch+1))
            netD_out_path = os.path.join(opt.out, 'netD_epoch_{}.pth'.format(epoch+1))
            netH_out_path = os.path.join(opt.out, 'netH_epoch_{}.pth'.format(epoch+1))
            torch.save(cut_netG.state_dict(), netG_out_path)
            torch.save(cut_netD.state_dict(), netD_out_path)
            torch.save(cut_netH.state_dict(), netH_out_path)