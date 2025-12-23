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
adversarial_loss = PatchAdversarialLoss(criterion="bce") # bce, least_squares
gpu_device = torch.device(f'cuda:{0}')
from generative.networks.nets import MultiScalePatchDiscriminator

adversarial_loss = PatchAdversarialLoss(criterion="bce") # bce, least_squares
    
def identity_loss(original_images, translated_images):
    loss = F.l1_loss(translated_images, original_images)
    return loss# Adversarial loss for generators

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels, num_filters=64, num_layers=3, strides=None):
        super(PatchGANDiscriminator, self).__init__()
        print('(PatchGANDiscriminator) %d layers - %d filters' % (num_layers, num_filters))

        if strides is None:
            strides = [2] * (num_layers - 2) + [1, 1]  # Default to [2, 2, 2, 1, 1] if strides are not provided

        self.layers = nn.ModuleList()

        # Initial convolution layer
        self.layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=strides[0], padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate convolution layers
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Conv2d(num_filters * 2**(i-1), num_filters * 2**i, kernel_size=4, stride=strides[i], padding=1))
            self.layers.append(nn.InstanceNorm2d(num_filters * 2**i))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Output layer
        self.layers.append(nn.Conv2d(num_filters * 2**(num_layers-2), 1, kernel_size=4, stride=strides[-1], padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels, num_d=3, num_filters=64, num_layers_d=3):
        super(MultiScaleDiscriminator, self).__init__()
        print('(MultiScaleDiscriminator) %d discriminators - %d layers - %d filters' % (num_d, num_layers_d, num_filters))
        self.num_d = num_d
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.discriminators = nn.ModuleList()
        
        # Create multiple discriminators
        for _ in range(num_d):
            self.discriminators.append(PatchGANDiscriminator(in_channels, num_filters, num_layers_d))
    
    def forward(self, x):
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
            # Downsample for the next discriminator
            x = self.downsample(x)
        return outputs  # list of outputs from all discriminators
    
    

class MultiScaleDiscriminatorStrided(nn.Module):
    def __init__(self, in_channels, num_d=3, num_filters=64, num_layers_d=3, downsample_filters=64):
        super(MultiScaleDiscriminatorStrided, self).__init__()
        self.num_d = num_d
        # Initial discriminator layers
        self.discriminators = nn.ModuleList()
        
        for _ in range(num_d):
            self.discriminators.append(PatchGANDiscriminator(in_channels, num_filters, num_layers_d))
            # After the first discriminator, increase the in_channels for subsequent ones
            # to match the output channels of the downsampling layer
            in_channels = downsample_filters
        
        # Learnable downsampling
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, downsample_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        outputs = []
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:  # Apply downsampling for all but the first discriminator
                x = self.downsample(x)
            outputs.append(discriminator(x))
        return outputs  # List of outputs from all discriminators

    
class MONAI_UNet(nn.Module):
    def __init__(self, num_res_units=6):
        super(MONAI_UNet, self).__init__()
        self.unet = nets.Unet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512, 512, 512, 512, 512),
            strides=(2, 2, 2, 2, 2, 2, 2),
            num_res_units=num_res_units,
        ).to(gpu_device)
        
    def forward(self, x):
        return self.unet(x)

class MONAI_UNet512(nn.Module):
    def __init__(self, num_res_units=6):
        super(MONAI_UNet512, self).__init__()
        self.unet = nets.Unet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512, 768, 768, 1024, 1024, 1024),
            strides=(1, 2, 2, 2, 2, 2, 2, 1),
            num_res_units=num_res_units,
        ).to(gpu_device)
        
    def forward(self, x):
        return self.unet(x)

class MONAI_UNet128(nn.Module):
    def __init__(self, num_res_units=6):
        super(MONAI_UNet128, self).__init__()
        self.unet = nets.Unet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512, 512, 512, 512),
            strides=(2, 2, 2, 2, 2, 2),
            num_res_units=num_res_units,
        ).to(gpu_device)
        
    def forward(self, x):
        return self.unet(x)   
    
class Pix2Pix(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9,
                 num_layers_d=3,
                ):
        super(Pix2Pix, self).__init__()
        self.generator_A_to_B = MONAI_UNet(num_res_units=num_res_units_G).to(gpu_device)
        self.discriminator_B = MultiScaleDiscriminator(in_channels=1, num_d=num_d, num_filters=num_filters_d, num_layers_d=num_layers_d ).to(gpu_device)
        self.criterionL1 = torch.nn.L1Loss()
        

    def calculate_edge_loss(self, img1, img2, alpha_NGF):
        from kornia.filters import SpatialGradient
        grad_src = SpatialGradient()(img1)
        grad_tgt = SpatialGradient()(img2)

        src_x = grad_src[:,:,0,:,:]
        src_y = grad_src[:,:,1,:,:]
        tgt_x = grad_tgt[:,:,0,:,:]
        tgt_y = grad_tgt[:,:,1,:,:]

        gradmag_src = torch.sqrt(torch.pow(src_x,2)+torch.pow(src_y,2)+alpha_NGF**2)
        gradmag_tgt = torch.sqrt(torch.pow(tgt_x,2)+torch.pow(tgt_y,2)+alpha_NGF**2)
        eps = 1e-8
        NGF = 1-1/2*(torch.pow((src_x/(gradmag_src+eps)*tgt_x/(gradmag_tgt+eps) + src_y/(gradmag_src+eps)*tgt_y/(gradmag_tgt+eps)),2))

        NGFM = torch.mean(NGF)

        return NGFM
    
    
    def compute_l1_loss(self, fake_B, real_B):
        l1_loss = self.criterionL1(real_B, fake_B)        
        return l1_loss  

    def compute_adv_loss(self, pred_fake_B):
        adv_loss = adversarial_loss(pred_fake_B, target_is_real=True, for_discriminator=False)  
        return adv_loss  
        

    def compute_NGF_loss(self, fake_B, real_B, alpha_NGF):
        NGF_loss = self.calculate_edge_loss(real_B, fake_B, alpha_NGF)
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

    
class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters_d=64, lambda_gan=1, lambda_identity=0.5, lambda_cyc=100):
        super(CycleGAN, self).__init__()
        self.lambda_gan=lambda_gan
        self.lambda_cyc=lambda_cyc
        self.lambda_identity=lambda_identity
        self.G_AB = MONAI_UNet().to(gpu_device)
        self.G_BA = MONAI_UNet().to(gpu_device)
        self.D_B = PatchGANDiscriminator(in_channels=1, num_filters=num_filters_d).to(gpu_device)
        self.D_A = PatchGANDiscriminator(in_channels=1, num_filters=num_filters_d).to(gpu_device)
        self.criterionL1 = torch.nn.L1Loss()

       
    def compute_generator_loss(self, fake_B, real_B, fake_A, real_A, cyc_B, cyc_A):
        # Adversarial loss for generators
        pred_fake_B = self.D_B(fake_B)
        pred_fake_A = self.D_B(fake_A)
        
        adv_loss_B= adversarial_loss(pred_fake_B, target_is_real=True, for_discriminator=False)
        adv_loss_A= adversarial_loss(pred_fake_A, target_is_real=True, for_discriminator=False)

        cyc_loss_B = self.criterionL1(real_B, cyc_B)
        cyc_loss_A = self.criterionL1(real_A, cyc_A)
        
        # Identity loss (optional)
        identity_B = self.G_AB(real_B)
        identity_loss_B = self.criterionL1(real_B, identity_B)
        identity_A = self.G_BA(real_A)
        identity_loss_A = self.criterionL1(real_A, identity_A)
        
        return cyc_loss_B, identity_loss_B, adv_loss_B, cyc_loss_A, identity_loss_A, adv_loss_A 
    
    def compute_discriminator_loss(self, real_B, fake_B, real_A, fake_A):
        # Adversarial loss for discriminators
        pred_real_B = self.D_B(real_B)
        pred_fake_B = self.D_B(fake_B.detach())  # Detach fake_B from the computation graph

        D_B_loss_real = adversarial_loss(pred_real_B, target_is_real=True, for_discriminator=True)
        D_B_loss_fake = adversarial_loss(pred_fake_B, target_is_real=False, for_discriminator=True)
        
        pred_real_A = self.D_A(real_A)
        pred_fake_A = self.D_A(fake_A.detach())  # Detach fake_A from the computation graph

        D_A_loss_real = adversarial_loss(pred_real_A, target_is_real=True, for_discriminator=True)
        D_A_loss_fake = adversarial_loss(pred_fake_A, target_is_real=False, for_discriminator=True)
        
        total_discriminator_loss = (
            D_B_loss_real
            + D_B_loss_fake
            + D_A_loss_real
            + D_A_loss_fake
        )
        return total_discriminator_loss

    def forward(self, real_A, real_B=None, is_training=True):
        # Translate images from domain A to domain B
        fake_B = self.G_AB(real_A)
        cyc_A =  self.G_BA(fake_B)
        
        fake_A = self.G_BA(real_B)
        cyc_B =  self.G_AB(fake_A)        
        # # Identity mapping (optional)
        if is_training:
            identity_B = self.G_AB(real_B)
            identity_A = self.G_BA(real_A)

        # Adversarial outputs
        pred_fake_B = self.D_B(fake_B)
        pred_fake_A = self.D_A(fake_A)

        # return fake_B, fake_A, identity_A, identity_B, pred_fake_A, pred_fake_B 
        if is_training:
            return fake_B, identity_B, pred_fake_B, fake_A, identity_A, pred_fake_A, cyc_B, cyc_A    
        else:
            return fake_B, pred_fake_B
        
# same as Pix2Pix class except that alpha_NGF is a trainable parameter
class Pix2PixNGF(Pix2Pix):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9,
                ):
        super(Pix2PixNGF, self).__init__(in_channels=in_channels, 
                                         out_channels=out_channels, 
                                         num_d=num_d, 
                                         num_filters_d=num_filters_d, 
                                         num_res_units_G=num_res_units_G)
        self.alpha_NGF = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        
# SD for Strided Discriminator
class Pix2PixSD(Pix2Pix):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9,
                 num_layers_d=3,
                 # You might add additional parameters specific to the new discriminator here
                ):
        # Initialize the parent class with all the required arguments
        super(Pix2PixSD, self).__init__(in_channels=in_channels, 
                                              out_channels=out_channels, 
                                              num_d=num_d, 
                                              num_filters_d=num_filters_d, 
                                              num_res_units_G=num_res_units_G,
                                              num_layers_d=num_layers_d)
        
        # Here you redefine the discriminator_B with the new discriminator you wish to use
        self.discriminator_B = MultiScaleDiscriminatorStrided(in_channels=1, 
                                                num_d=num_d, 
                                                num_filters=num_filters_d, 
                                                num_layers_d=num_layers_d).to(gpu_device)
class Pix2Pix512(Pix2Pix):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9,  # Adjust this if needed for your MONAI_UNet512
                 num_layers_d=3,
                ):
        super(Pix2Pix512, self).__init__(in_channels=in_channels, 
                                         out_channels=out_channels, 
                                         num_d=num_d, 
                                         num_filters_d=num_filters_d, 
                                         num_res_units_G=num_res_units_G, 
                                         num_layers_d=num_layers_d)

        # Replace the generator with MONAI_UNet512
        self.generator_A_to_B = MONAI_UNet512(num_res_units=num_res_units_G).to(gpu_device)
        # Assuming discriminator and other components remain unchanged

class Pix2Pix128(Pix2Pix):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_d=1, 
                 num_filters_d=64, 
                 num_res_units_G=9,  # Adjust this if needed for your MONAI_UNet512
                 num_layers_d=3,
                ):
        super(Pix2Pix128, self).__init__(in_channels=in_channels, 
                                         out_channels=out_channels, 
                                         num_d=num_d, 
                                         num_filters_d=num_filters_d, 
                                         num_res_units_G=num_res_units_G, 
                                         num_layers_d=num_layers_d)

        # Replace the generator with MONAI_UNet512
        self.generator_A_to_B = MONAI_UNet128(num_res_units=num_res_units_G).to(gpu_device)
        # Assuming discriminator and other components remain unchanged