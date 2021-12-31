
import models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math

import sys
from PIL import Image
import torchvision
import argparse
import random
from utils import adjust_scales2image, generate_noise2, calc_gradient_penalty
from imresize import imresize2
import os.path as osp
import torchvision.utils as vutils
from torch.utils.data import  DataLoader
from tqdm import tqdm
import cut.hyperparameters as hyperparams
import cut.dataset as dataset
import cut.cut_model as cut_model
import cut.cut_train as cut_train

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
def draw_concat(Gs,reals, NoiseAmp, in_s, mode, opt):
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            for G,real_curr,real_next,noise_amp in zip(Gs,reals,reals[1:],NoiseAmp):
                G = G.cuda()
                if count == 0:
                    z = generate_noise2([1, 3, real_curr.shape[2], real_curr.shape[3]], device=opt.device)
                    G_z = in_s
                else:
                    z = generate_noise2([1, opt.nc_z,real_curr.shape[2], real_curr.shape[3]], device=opt.device)

                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                z_in = noise_amp*z+G_z
                if count > opt.switch_scale:
                    G_z = G(z_in.detach())
                else:
                    G_z = G(z_in.detach(), G_z)
                G_z = imresize2(G_z.detach(),1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1

        if mode == 'rec':
            count = 0
            for G,real_curr,real_next,noise_amp in zip(Gs,reals,reals[1:],NoiseAmp):
                G = G.cuda()
                if count == 0:
                    size = list(real_curr.size())
                    #print(size)
                    G_z = generate_noise2(size, device=opt.device)
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                if count > opt.switch_scale:
                    G_z = G(G_z)
                else:
                    G_z = G(G_z, G_z)
                G_z = imresize2(G_z.detach(), 1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
    return G_z


def init_models(opt):

    #generator initialization:
    netG = models.Generator_no_res(opt).to(opt.device)
    netG.apply(models.weights_init)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)

    return netD, netG


def init_models_res(opt):
    # generator initialization:
    netG = models.Generator(opt).to(opt.device)
    netG.apply(models.weights_init)

    # discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)

    return netD, netG


def transform_input(img_path, opt):

    res = []
    image = Image.open(img_path).convert('RGB')
    for ii in range(0, opt.stop_scale + 1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - ii)

        s_size = math.ceil(scale * opt.img_size)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((s_size, s_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        sample = transform(image)
        res.append(sample.unsqueeze(0))

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id, if the value is -1, the cpu is used')
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)

    # load, input, save configurations:
    parser.add_argument('--load', default='', help="path to continue training")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z', type=int, help='noise # channels', default=3)
    parser.add_argument('--nc_im', type=int, help='image # channels', default=3)

    # networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)
    parser.add_argument('--num_layer', type=int, help='number of layers', default=5)
    parser.add_argument('--stride', help='stride', default=1)
    parser.add_argument('--padd_size', type=int, help='net pad size', default=0)

    # pyramid parameters:
    parser.add_argument('--scale_factor', type=float, help='pyramid scale factor', default=0.75)
    parser.add_argument('--noise_amp_a', type=float, help='addative noise cont weight', default=0.1)
    parser.add_argument('--noise_amp_b', type=float, help='addative noise cont weight', default=0.1)
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=18)
    parser.add_argument('--max_size', type=int, help='image minimal size at the coarser scale', default=250)

    # optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=20000, help='number of epochs to train per scale')
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0001, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lambda_grad', type=float, help='gradient penelty weight', default=0.1)
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=1.0)
    parser.add_argument('--beta', type=float, help='cycle loss weight', default=1.0)
    parser.add_argument('--lambda_g', type=float, default=1.0, help='change ratio between gan loss, multiply by the gan loss of image B')

    #main arguments
    parser.add_argument('--input_a', help='input image path', required=True)
    parser.add_argument('--input_b', help='input image path', required=True)
    parser.add_argument('--switch_res', type=int, default=2, help='how many levels will not be residual')
    parser.add_argument('--img_size', type=int, default=220, help='image size of the output')
    parser.add_argument('--out', required=True)
    parser.add_argument('--print_interval', type=int, default=1000)
    opt = parser.parse_args()

    if not os.path.exists(opt.out):
        os.makedirs(opt.out)

    torch.cuda.set_device(opt.gpu_id)

    opt.device = "cuda:%s" % opt.gpu_id
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp_a
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor

    adjust_scales2image(opt.img_size, opt)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.gpu_id == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    Gs_a = []
    reals_a = []
    NoiseAmp_a = []
    
    Gs_b = []
    reals_b = []
    NoiseAmp_b = []
    
    nfc_prev = 0
    scale_num = 0

    r_loss = nn.MSELoss()

    data_a = transform_input(opt.input_a, opt)
    data_b = transform_input(opt.input_b, opt)

    size_arr = []
    for ii in range(0, opt.stop_scale + 1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - ii)
        size_arr.append(math.ceil(scale * opt.img_size))

    opt.switch_scale = opt.stop_scale - opt.switch_res

    opt.nzx = size_arr[0]
    opt.nzy = size_arr[0]
    in_s = torch.full([1, opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)

    if opt.load != '':
        Gs_a = torch.load('%s/Gs_a.pth' % opt.load)
        Gs_b = torch.load('%s/Gs_b.pth' % opt.load)
        NoiseAmp_a = torch.load('%s/NoiseAmp_a.pth' % opt.load)
        NoiseAmp_b = torch.load('%s/NoiseAmp_b.pth' % opt.load)
        scale_num = len(Gs_a)
        opt.noise_amp_a = NoiseAmp_a[-1]
        opt.noise_amp_b = NoiseAmp_b[-1]
        print("Loading until scale " + str(scale_num))
        nfc_prev = min(opt.nfc_init * pow(2, math.floor((scale_num-1) / 4)), 128)
    else:
        opt.load = opt.out
    
    while scale_num < opt.stop_scale + 1:

        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        if scale_num > opt.switch_scale:
            D_a, G_a = init_models(opt)
            D_b, G_b = init_models(opt)
            print("No Residual layer")
        else:
            D_a, G_a = init_models_res(opt)
            D_b, G_b = init_models_res(opt)
            print("Residual layer")
        
        #==========CUT============
        cut_netG = cut_model.Generator(input_shape=(opt.nc_z,opt.nzx,opt.nzy)).to(opt.device)
        cut_model.weights_init(cut_netG)

        cut_netD = cut_model.Discriminator(input_shape=(opt.nc_z,opt.nzx,opt.nzy)).to(opt.device)
        cut_model.weights_init(cut_netD)

        cut_netH = cut_model.MLP().to(opt.device)
        cut_model.weights_init(cut_netH)

        cut_criterionBCE = nn.BCELoss().to(opt.device)
        cut_criterionMSE = nn.MSELoss().to(opt.device)

        cut_optimizerD = optim.Adam(cut_netD.parameters(), lr=hyperparams.lr, betas=(hyperparams.beta1, 0.999))
        cut_optimizerG = optim.Adam(cut_netG.parameters(), lr=hyperparams.lr, betas=(hyperparams.beta1, 0.999))
        cut_optimizerH = optim.Adam(cut_netH.parameters(), lr=hyperparams.lr, betas=(hyperparams.beta1, 0.999))
        #==========CUT============

        if nfc_prev == opt.nfc:
            print("Load weights of last layer " + str(scale_num-1))
            G_a.load_state_dict(torch.load('%s/netG_a_%d.pth' % (opt.load, scale_num-1)))
            D_a.load_state_dict(torch.load('%s/netD_a_%d.pth' % (opt.load, scale_num-1)))
            G_b.load_state_dict(torch.load('%s/netG_b_%d.pth' % (opt.load, scale_num-1)))
            D_b.load_state_dict(torch.load('%s/netD_b_%d.pth' % (opt.load, scale_num-1)))
            #CUT
            cut_netG.load_state_dict(torch.load('%s/cut_netG_%d.pth' % (opt.load, scale_num-1)))
            cut_netD.load_state_dict(torch.load('%s/cut_netD_%d.pth' % (opt.load, scale_num-1)))
            cut_netH.load_state_dict(torch.load('%s/cut_netH_%d.pth' % (opt.load, scale_num-1)))
            
        opt.load = opt.out

        optimizerD = optim.Adam(list(D_a.parameters()) + list(D_b.parameters()), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(list(G_a.parameters()) + list(G_b.parameters()), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        n_iters = opt.niter

        opt.nzx = size_arr[len(Gs_a)]
        opt.nzy = size_arr[len(Gs_a)]

        noise_amount_a = 0
        noise_cnt_a = 0

        noise_amount_b = 0
        noise_cnt_b = 0

        i = 0
        for epoch in tqdm(range(n_iters)):

            real_a = data_a[len(Gs_a)].cuda()

            real_b = data_b[len(Gs_b)].cuda()

            noise_ = generate_noise2([1, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)

            if Gs_a == []:
                noise_a = noise_
                prev_a = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
            else:
                prev_a = draw_concat(Gs_a,list(data_a), NoiseAmp_a, in_s, 'rand', opt)
                noise_a = opt.noise_amp_a * noise_ + prev_a

            noise_ = generate_noise2([1, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)

            if Gs_b == []:
                noise_b = noise_
                prev_b = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
            else:
                prev_b = draw_concat(Gs_b,list(data_b), NoiseAmp_b, in_s, 'rand', opt)
                noise_b = opt.noise_amp_b * noise_ + prev_b

            if scale_num > opt.switch_scale:
                fake_a = G_a(noise_a.detach())
                fake_b = G_b(noise_b.detach())
            else:
                fake_a = G_a(noise_a.detach(), prev_a.detach())
                fake_b = G_b(noise_b.detach(), prev_b.detach())

            if Gs_a == []:
                z_prev_a = generate_noise2([1, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
            else:
                z_prev_a = draw_concat(Gs_a,list(data_a), NoiseAmp_a, in_s, 'rec', opt)

            if epoch == 0 and i == 0:
                if Gs_a == []:
                    opt.noise_amp_a = opt.noise_amp_init
                else:
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real_a, z_prev_a))
                    opt.noise_amp_a = opt.noise_amp_init * RMSE

            if Gs_b == []:
                z_prev_b = generate_noise2([1, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
            else:
                z_prev_b = draw_concat(Gs_b,list(data_b), NoiseAmp_b, in_s, 'rec', opt)

            if epoch == 0 and i == 0:
                if Gs_b == []:
                    opt.noise_amp_b = opt.noise_amp_init
                else:
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real_b, z_prev_b))
                    opt.noise_amp_b = opt.noise_amp_init * RMSE

            i += 1

            if scale_num > opt.switch_scale:
                generated_a = G_a(z_prev_a.detach())
                generated_b = G_b(z_prev_b.detach())
            else:
                generated_a = G_a(z_prev_a.detach(), z_prev_a.detach())
                generated_b = G_b(z_prev_b.detach(), z_prev_b.detach())

            if scale_num > opt.switch_scale:
                mix_g_a = G_a(fake_b)
                mix_g_b = G_b(fake_a)
            else:
                mix_g_a = G_a(fake_b, fake_b)
                mix_g_b = G_b(fake_a, fake_a)

            other_noise_a = generate_noise2([1, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
            other_noise_b = generate_noise2([1, opt.nc_z, opt.nzx, opt.nzy], device=opt.device)

            noisy_real_b = opt.noise_amp_a * other_noise_a + real_b
            noisy_real_a = opt.noise_amp_b * other_noise_b + real_a


            #############################
            ####      Train D_a      ####
            #############################

            D_a.zero_grad()

            output = D_a(real_a).to(opt.device)
            errD_real = -2 * output.mean()  # -a
            errD_real.backward(retain_graph=True)

            output_a = D_a(mix_g_a.detach())
            output_a2 = D_a(fake_a.detach())
            errD_fake_a = output_a.mean() + output_a2.mean()
            errD_fake_a.backward(retain_graph=True)

            gradient_penalty_a = calc_gradient_penalty(D_a, real_a, mix_g_a, opt.lambda_grad, opt.device)
            gradient_penalty_a += calc_gradient_penalty(D_a, real_a, fake_a, opt.lambda_grad, opt.device)
            gradient_penalty_a.backward(retain_graph=True)

            #############################
            ####      Train D_b      ####
            #############################

            D_b.zero_grad()

            output = D_b(real_b).to(opt.device)
            errD_real = -2 * output.mean()  # -a
            errD_real.backward(retain_graph=True)

            output_b = D_b(mix_g_b.detach())
            output_b2 = D_b(fake_b.detach())
            errD_fake_b = output_b.mean() + output_b2.mean()
            errD_fake_b.backward(retain_graph=True)

            gradient_penalty_b = calc_gradient_penalty(D_b, real_b, mix_g_b, opt.lambda_grad, opt.device)
            gradient_penalty_b += calc_gradient_penalty(D_b, real_b, fake_b, opt.lambda_grad, opt.device)
            gradient_penalty_b.backward(retain_graph=True)

            optimizerD.step()

            #############################
            ####      Train G      ####
            #############################

            G_a.zero_grad()
            G_b.zero_grad()

            output_a = D_a(mix_g_a)
            output_a2 = D_a(fake_a)
            errG_a = -output_a.mean() -output_a2.mean()
            errG_a.backward(retain_graph=True)

            output_b = D_b(mix_g_b)
            output_b2 = D_b(fake_b)
            errG_b = opt.lambda_g * (-output_b.mean() -output_b2.mean())
            errG_b.backward(retain_graph=True)

            if opt.alpha > 0:
                rec_loss_a = opt.alpha * r_loss(generated_a, real_a)
                rec_loss_a.backward(retain_graph=True)

                rec_loss_b = opt.alpha * r_loss(generated_b, real_b)
                rec_loss_b.backward(retain_graph=True)

            #cutData=dataset.SingleImageDataset(fake_a.cpu().detach(),fake_b.cpu().detach())
            #train_dataloader = DataLoader(cutData, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.workers)
            # print(cutData[20]['A'].shape)
            #============CUT==============
            ############################
            # (1) Update D network
            ###########################
            real_label = 1.0
            fake_label = 0.
            cut_netG.train()
            cut_netD.train()
            cut_netH.train()
            ## Train with all-real batch
            cut_netD.zero_grad()
            # Format batch
            real = real_b.to(opt.device)
            g_input = fake_b.to(opt.device)
            # print(real.shape)
            '''
            '''
            pred_real = cut_netD(real).view(-1) # view: reshape, -1: uncertion columns number
            label = torch.full_like(pred_real, real_label, dtype=torch.float, device=opt.device) # size[900] (30 * 30 patches)
            errD_real = cut_criterionMSE(pred_real, label)

            ## Train with all-fake batch
            '''
            '''
            fake = cut_netG(g_input)
            # print("=======")
            # print(g_input.shape)
            # print(fake.shape)
            # print("=======")
            # detach: The result will never require gradient.
            pred_fake = cut_netD(fake.detach()).view(-1)
            label = torch.full_like(pred_fake, fake_label, dtype=torch.float, device=opt.device)
            errD_fake = cut_criterionMSE(pred_fake, label)

            errD = (errD_real + errD_fake) * 0.5
            errD.backward()
            cut_optimizerD.step()
            ###########################
            # (2) Update G network
            ###########################
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
                fk, sample_list = cut_netH(featk, enc_layer=nce_layers[i])
                fq, _ = cut_netH(featq,sample_list, enc_layer=nce_layers[i])
                #cut_errG_PatchNCE += cut_model.PatchNCELoss(fq, fk, opt.device)
            cut_errG_PatchNCE /= len(nce_layers)
            
            # Calculate identity loss (multi-layer)
            # cut_errG_idt = 0.0
            # fake_y = cut_netG(real)
            # fqs_enc = cut_netG(fake_y, enc_layer=nce_layers)
            # fks_enc = cut_netG(real, enc_layer=nce_layers)
            # for i, (featq, featk) in enumerate(zip(fqs_enc, fks_enc)):
            #     fk, sample_list = cut_netH(featk, enc_layer=nce_layers[i])
            #     fq, _ = cut_netH(featq,sample_list, enc_layer=nce_layers[i])
            #     cut_errG_idt += cut_model.PatchNCELoss(fq, fk, opt.device)
            # cut_errG_idt /= len(nce_layers)

            # Sum the loss and backward
            # cut_errG = cut_errG_GAN + 0.5 * (hyperparams.lamb_x * cut_errG_PatchNCE + hyperparams.lamb_y * cut_errG_idt)
            cut_errG = cut_errG_GAN + 0.5 * (10 * cut_errG_PatchNCE)
            cut_errG.backward()
            cut_optimizerG.step()
            cut_optimizerH.step()

            #============CUT==============
            # if opt.beta > 0:
            #     if scale_num > opt.switch_scale:
            #         cycle_a = G_a(mix_g_b)
            #     else:
            #         cycle_a = G_a(mix_g_b, mix_g_b)

            #     cycle_loss_a = opt.beta * r_loss(cycle_a, fake_a)
            #     cycle_loss_a.backward(retain_graph=True)

            # if opt.beta > 0:
            #     if scale_num > opt.switch_scale:
            #         cycle_b = G_b(mix_g_a)
            #     else:
            #         cycle_b = G_b(mix_g_a, mix_g_a)

            #     cycle_loss_b = opt.beta * r_loss(cycle_b, fake_b)
            #     cycle_loss_b.backward(retain_graph=True)

            optimizerG.step()

            if (epoch+1) % opt.print_interval == 0:
                vutils.save_image(fake_a.clone(), osp.join(opt.out, str(scale_num) + "_fake_a_" + str(epoch) + ".png"), normalize=True)
                vutils.save_image(mix_g_a.clone(), osp.join(opt.out, str(scale_num) + "_b2a_" + str(epoch) + ".png"),
                                  normalize=True)

                if epoch == 0:
                    vutils.save_image(real_a.clone(), osp.join(opt.out, str(scale_num) + "_real_a_" + str(epoch) + ".png"), normalize=True)

                vutils.save_image(fake_b.clone(), osp.join(opt.out, str(scale_num) + "_fake_b_" + str(epoch) + ".png"),
                                  normalize=True)
                vutils.save_image(mix_g_b.clone(), osp.join(opt.out, str(scale_num) + "_a2b_" + str(epoch) + ".png"),
                                  normalize=True)
                if epoch == 0:
                    vutils.save_image(real_b.clone(), osp.join(opt.out, str(scale_num) + "_real_b_" + str(epoch) + ".png"), normalize=True)

                print("debug imgs saved, scale_num=%0d, epoch=%0d " % (scale_num, epoch))
                sys.stdout.flush()

        if scale_num == opt.stop_scale:
            vutils.save_image(fake_a.clone(), osp.join(opt.out,  "final_fake_a_" + str(epoch) + ".png"),
                              normalize=True)
            vutils.save_image(mix_g_a.clone(), osp.join(opt.out, "final_b2a_" + str(epoch) + ".png"),
                              normalize=True)

            vutils.save_image(fake_b.clone(), osp.join(opt.out, "final_fake_b_" + str(epoch) + ".png"),
                              normalize=True)
            vutils.save_image(mix_g_b.clone(), osp.join(opt.out, "final_a2b_" + str(epoch) + ".png"),
                              normalize=True)

        Gs_a.append(G_a)
        NoiseAmp_a.append(opt.noise_amp_a)

        torch.save(Gs_a, '%s/Gs_a.pth' % (opt.out))
        torch.save(reals_a, '%s/reals_a.pth' % (opt.out))
        torch.save(NoiseAmp_a, '%s/NoiseAmp_a.pth' % (opt.out))

        torch.save(G_a.state_dict(), '%s/netG_a_%d.pth' % (opt.out, scale_num))
        torch.save(D_a.state_dict(), '%s/netD_a_%d.pth' % (opt.out, scale_num))

        Gs_b.append(G_b)
        NoiseAmp_b.append(opt.noise_amp_b)

        torch.save(Gs_b, '%s/Gs_b.pth' % (opt.out))
        torch.save(reals_b, '%s/reals_b.pth' % (opt.out))
        torch.save(NoiseAmp_b, '%s/NoiseAmp_b.pth' % (opt.out))

        torch.save(G_b.state_dict(), '%s/netG_b_%d.pth' % (opt.out, scale_num))
        torch.save(D_b.state_dict(), '%s/netD_b_%d.pth' % (opt.out, scale_num))

        #CUT
        torch.save(cut_netG.state_dict(),  '%s/cut_netG_%d.pth' % (opt.out, scale_num))
        torch.save(cut_netD.state_dict(),  '%s/cut_netD_%d.pth' % (opt.out, scale_num))
        torch.save(cut_netH.state_dict(),  '%s/cut_netH_%d.pth' % (opt.out, scale_num))

        print("Layer weights saved successfully")

        scale_num += 1
        nfc_prev = opt.nfc
        del D_a, G_a
        del D_b, G_b





