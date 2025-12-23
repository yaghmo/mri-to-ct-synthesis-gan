#  Vincent Jaouen' I2I networks with MONAI
#  The generator is MONAI's UNet
#  The discriminator is a multiscale conv encoder (à la pix2pixHD)
#  vincent.jaouen@imt-atlantique.fr

import torch 
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from monai.networks.blocks import ResBlock
from generative.losses import PatchAdversarialLoss
import monai.networks.nets as nets
from generative.networks.nets import MultiScalePatchDiscriminator

gpu_device = torch.device(f'cuda:{0}')
adversarial_loss = PatchAdversarialLoss(criterion="bce") # bce, least_squares

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels, num_filters=64, num_layers=3, strides=None):
        super(PatchGANDiscriminator, self).__init__()

        if strides is None:
            strides = [2] * (num_layers - 2) + [1, 1]  # Default to [2, 2, 2, 1, 1] if not provided

        self.layers = nn.ModuleList()

        # Initial convolution layer
        self.layers.append(nn.Conv3d(in_channels, num_filters, kernel_size=4, stride=strides[0], padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate convolution layers
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Conv3d(num_filters * 2**(i-1), num_filters * 2**i, kernel_size=4, stride=strides[i], padding=1))
            self.layers.append(nn.InstanceNorm3d(num_filters * 2**i))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Output layer
        self.layers.append(nn.Conv3d(num_filters * 2**(num_layers-2), 1, kernel_size=4, stride=strides[-1], padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels, num_d=2, num_filters=64, num_layers_d=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_d = num_d
        self.downsample = nn.AvgPool3d(3, stride=2, padding=[1, 1, 1], count_include_pad=False)
        self.discriminators = nn.ModuleList()

        for _ in range(num_d):
            self.discriminators.append(PatchGANDiscriminator(in_channels, num_filters, num_layers_d))
    
    
    def forward(self, x):
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
            # Downsample for the next discriminator
            x = self.downsample(x)
        return outputs  # list of outputs from all discriminators

class MONAI_UNet3d(nn.Module):
    def __init__(self, num_res_units=6):
        super(MONAI_UNet3d, self).__init__()
        self.unet = nets.Unet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels = (64,128,128,256,512,512,512), 
            strides = (2,2,2,2,2,2),
            # channels=(64, 128, 256, 512, 512, 512, 512, 512),
            # strides=(2, 2, 2, 2, 2, 2, 2),
            num_res_units=num_res_units,
        ).to(gpu_device)
        
    def forward(self, x):
        return self.unet(x)

    
    
class Pix2Pix_3d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_layers_d=1, 
                 num_d=2, 
                 num_filters_d=64, 
                 lambda_gan=1, 
                 lambda_identity=0.5, 
                 lambda_l1=100, 
                 lambda_NGF=1, 
                 num_res_units_G=9):
        super(Pix2Pix_3d, self).__init__()
        self.generator_A_to_B = MONAI_UNet3d(num_res_units=num_res_units_G).to(gpu_device)
        self.discriminator_B = MultiScaleDiscriminator(in_channels=1, num_d=2, num_filters=num_filters_d, num_layers_d=3).to(gpu_device)
        self.criterionL1 = torch.nn.L1Loss()

    def calculate_edge_loss(self, img1, img2, alpha_NGF=0.05):
        grad_src = SpatialGradient3d()(img1)
        grad_tgt = SpatialGradient3d()(img2)

        src_x = grad_src[:,:,:,0,:,:]
        src_y = grad_src[:,:,:,1,:,:]
        src_z = grad_src[:,:,:,2,:,:]
        tgt_x = grad_tgt[:,:,:,0,:,:]
        tgt_y = grad_tgt[:,:,:,1,:,:]
        tgt_z = grad_tgt[:,:,:,2,:,:]

        gradmag_src = torch.sqrt(torch.pow(src_x, 2) + torch.pow(src_y, 2) + torch.pow(src_z, 2) + alpha_NGF**2)
        gradmag_tgt = torch.sqrt(torch.pow(tgt_x, 2) + torch.pow(tgt_y, 2) + torch.pow(tgt_z, 2) + alpha_NGF**2)
        eps = 1e-8
        NGF = 1 - 1/2 * (torch.pow((src_x/(gradmag_src + eps) * tgt_x/(gradmag_tgt + eps) + 
                                    src_y/(gradmag_src + eps) * tgt_y/(gradmag_tgt + eps) + 
                                    src_z/(gradmag_src + eps) * tgt_z/(gradmag_tgt + eps)), 2))

        NGFM = torch.mean(NGF)
        return NGFM

    
    
    def compute_l1_loss(self, fake_B, real_B):
        l1_loss = self.criterionL1(real_B, fake_B)        
        return l1_loss  
    
    def compute_adv_loss(self, fake_B, real_B):
        adv_loss = adversarial_loss(pred_fake_B, target_is_real=True, for_discriminator=False)  
        return adv_loss  
    
    def compute_NGF_loss(self, fake_B, real_B):
        NGF_loss = self.calculate_edge_loss(real_B, fake_B)
        return NGF_loss         
    
    def compute_identity_loss(self, real_B):
        identity_B = self.generator_A_to_B(real_B)
        identity_loss =  self.criterionL1(real_B, identity_B)
        return identity_loss     
    
    def compute_discriminator_loss(self, real_B, fake_B):
        # Adversarial loss for discriminators
        pred_real_B = self.discriminator_B(real_B)
        pred_fake_B = self.discriminator_B(fake_B.detach())  # Detach fake_B from the computation graph

        discriminator_B_loss_real = adversarial_loss(pred_real_B, target_is_real=True, for_discriminator=True)
        discriminator_B_loss_fake = adversarial_loss(pred_fake_B, target_is_real=False, for_discriminator=True)

        total_discriminator_loss = (
            discriminator_B_loss_real
            + discriminator_B_loss_fake
        )
        return total_discriminator_loss

    def forward(self, real_A, real_B=None, is_training=True):
        # Translate images from domain A to domain B
        fake_B = self.generator_A_to_B(real_A)

        # # Identity mapping (optional)
        if is_training:
            identity_B = self.generator_A_to_B(real_B)

        # Adversarial outputs
        pred_fake_B = self.discriminator_B(fake_B)

        # return fake_B, fake_A, identity_A, identity_B, pred_fake_A, pred_fake_B 
        if is_training:
            return fake_B, identity_B, pred_fake_B    
        else:
            return fake_B, pred_fake_B