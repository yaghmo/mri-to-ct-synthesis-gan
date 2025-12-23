#  Vincent Jaouen' I2I networks with MONAI
#  The generator is MONAI's UNet
#  The discriminator is a multiscale conv encoder (à la pix2pixHD)
#  vincent.jaouen@imt-atlantique.fr

from .vjnetworks import Pix2Pix, CycleGAN, Pix2PixNGF, Pix2PixSD, Pix2Pix512, Pix2Pix128
from .vjnetworks3d import Pix2Pix_3d
