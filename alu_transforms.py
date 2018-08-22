
import sys
import json
import argparse
import numpy as np
from time import time
from PIL import Image
import math as math

from basenet import BaseNet
from basenet.lr import LRSchedule
from basenet.helpers import to_numpy, set_seeds

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True
import os
import PIL
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from torchvision import transforms, datasets


def save_image(img, file_name,save_pic = False):
    file_name = file_name + '.png'
    file_exist = os.path.exists(file_name)
    if (save_pic and file_exist == False):
        plt.imshow(img, interpolation='nearest')
        plt.savefig(file_name)

class AugmantationClass(object):
    def __init__(self):
        self.num_of_p = self.get_num_of_p()
        self.num_of_mag = self.get_num_of_mag()
        self.mag_scale_value = random.randint(0,self.num_of_mag - 1)
        possible_probs = np.linspace(start=0.1, stop=1, num=self.num_of_p)
        self.curr_p = random.choice(possible_probs)

    def get_num_of_p(self):
        return 10

    def get_num_of_mag(self):
        return 10

    def get_num_of_possible_values(self):
        return self.num_of_p * self.num_of_mag

    def do_aug(self, img, prob, scale_value):
        raise NotImplementedError('implemented by child')


class shearX(AugmantationClass):


    def do_aug(self, img,prob,scale_value):
        if random.random() < prob:
            save_image(img, 'before_shear_x')
            shear_value = np.linspace(start=-0.3, stop=0.3, num=10) #TODO it moves a lot
            shear_scale_to_set = shear_value[scale_value]
            ##transform ax + by + c, dx + ey + f
            a = 1
            b = shear_scale_to_set
            c = 0   # left/right (i.e. 5/-5)
            d = 0
            e = 1
            f = 0  # up/down (i.e. 5/-5)
            img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
            save_image(img, 'after_shear_x')
        return img

class shearY(AugmantationClass):
    def do_aug(self, img, prob, scale_value):
        if random.random() < prob:
            save_image(img, 'before_shear_y')
            shear_value = np.linspace(start=-0.3, stop=0.3, num=10) #TODO it moves a lot
            shear_scale_to_set = shear_value[scale_value]
            ##transform ax + by + c, dx + ey + f
            a = 1
            b = 0
            c = 0   # left/right (i.e. 5/-5)
            d = shear_scale_to_set
            e = 1
            f = 0  # up/down (i.e. 5/-5)
            img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
            save_image(img, 'after_shear_y')
        return img


class cutout(AugmantationClass):
    def do_aug(self, img, prob, scale_value):
        if random.random() < prob:
            save_image(img, 'before_cutout')
            cutout_value = np.linspace(start=2, stop=20, num=10) #the picture is 40 * 40 (32 + 4 + 4 - padding)
            length = cutout_value[scale_value]

            h = img.height
            w = img.width

            n_holes = 1 #TODO


            mask = np.ones((h, w), np.float32)

            for n in range(n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = int(np.clip(y - length // 2, 0, h))
                y2 = int(np.clip(y + length // 2, 0, h))
                x1 = int(np.clip(x - length // 2, 0, w))
                x2 = int(np.clip(x + length // 2, 0, w))

                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)

            trnsform_pil_to_tensor = transforms.ToTensor()
            img = trnsform_pil_to_tensor(img)

            mask = mask.expand_as(img)

            img = img * mask

            transform_to_pil = transforms.ToPILImage()
            img = transform_to_pil(img)

            save_image(img, 'after_cutout')

        return img


class translateX(AugmantationClass):
    def do_aug(self, img, prob, scale_value):
        if random.random() < prob:
            save_image(img, 'before_translate_x')
            translate_value = np.linspace(start=-15, stop=15, num=10) #TODO it moves a lot
            translate_scale_to_set = translate_value[scale_value]
            ##transform ax + by + c, dx + ey + f
            a = 1
            b = 0
            c = translate_scale_to_set   # left/right (i.e. 5/-5)
            d = 0
            e = 1
            f = 0  # up/down (i.e. 5/-5)
            img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
            save_image(img, 'after_translate_x')
        return img

class translateY(AugmantationClass):
    def do_aug(self, img, prob, scale_value):
        if random.random() < prob:
            save_image(img, 'before_translate_y')
            translate_value = np.linspace(start=-15, stop=15, num=10) #TODO it moves a lot
            translate_scale_to_set = translate_value[scale_value]
            ##transform ax + by + c, dx + ey + f
            a = 1
            b = 0
            c = 0   # left/right (i.e. 5/-5)
            d = 0
            e = 1
            f = scale_value  # up/down (i.e. 5/-5)
            img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
            save_image(img, 'after_translate_y')
        return img

class equalize(AugmantationClass):
    def get_num_of_mag(self):
        return 1

    def do_aug(self, img, prob, scale_value):
        if random.random() < prob:
            save_image(img, 'before_equalize')
            return PIL.ImageOps.equalize(img)
            save_image(img, 'after_equalize')
        else:
            return img

class invert(AugmantationClass):
    def get_num_of_mag(self):
        return 1

    def do_aug(self, img, prob, scale_value):
        if random.random() < prob:
            save_image(img, 'before_invert')
            img = PIL.ImageOps.invert(img)
            save_image(img, 'after_invert')

            return img
        else:
            return img

class color(AugmantationClass):
    def do_aug(self, img, prob, scale_value):
        color_value = np.linspace(start=0.1, stop=1.9, num=10)
        if random.random() < prob:
            save_image(img, 'before_color')
            color_scale_to_set = color_value[scale_value]
            color = PIL.ImageEnhance.Color(img)
            img = color.enhance(color_scale_to_set)
            save_image(img, 'after_color')
            #sys.exit(1)  # TODO
            return img
        else:
            return img

class autocontrast(AugmantationClass):
    def get_num_of_mag(self):
        return 1

    def do_aug(self, img, prob, scale_value):
        if random.random() < prob:

            save_image(img, 'before_autocontrast')
            img = PIL.ImageOps.autocontrast(img)
            save_image(img, 'after_autocontrast')
            return img
        else:
            return img

# class posterize(AugmantationClass):
#     def do_aug(self, img, prob, scale_value):
#         if random.random() < prob:
#             posterize_value = np.linspace(start=4, stop=8, num=5)
#             save_image(img, 'before_posterize')
#             posterize_scale_to_set = posterize_value[scale_value]
#             img = PIL.ImageOps.posterize(img,bits = posterize_scale_to_set)
#             save_image(img, 'after_posterize')
#             return img
#         else:
#             return img


class brightness(AugmantationClass):
    def do_aug(self, img, prob, scale_value):
        brightness_value = np.linspace(start=0.1, stop=1.9, num=10)
        if random.random() < prob:
            save_image(img, 'before_brightness')
            brightness_scale_to_set = brightness_value[scale_value]
            brightness = PIL.ImageEnhance.Brightness(img)
            img = brightness.enhance(brightness_scale_to_set)
            save_image(img, 'after_brightness')
            return img
        else:
            return img

class contrast(AugmantationClass):
    def do_aug(self, img, prob, scale_value):
        contrast_value = np.linspace(start=0.1, stop=1.9, num=10)
        if random.random() < prob:
            save_image(img, 'before_contrast')
            contrast_scale_to_set = contrast_value[scale_value]
            contrast = PIL.ImageEnhance.Contrast(img)
            img = contrast.enhance(contrast_scale_to_set)
            save_image(img, 'after_contrast')
            return img
        else:
            return img

class sharpness(AugmantationClass):
    def do_aug(self, img, prob, scale_value):
        sharpness_value = np.linspace(start=0.1, stop=1.9, num=10)
        if random.random() < prob:
            save_image(img, 'before_sharpness')
            sharpness_scale_to_set = sharpness_value[scale_value]
            sharpness = PIL.ImageEnhance.Sharpness(img)
            img = sharpness.enhance(sharpness_scale_to_set)
            save_image(img, 'after_sharpness')
            return img

        else:
            return img

class rotate(AugmantationClass):
    def do_aug(self, img, prob, scale_value):
        rotate_value = np.linspace(start= -30, stop=30, num=10)
        if random.random() < prob:
            save_image(img, 'before_rotate')
            angle = rotate_value[scale_value]
            img = img.rotate(angle)
            save_image(img, 'after_rotate')

            return img
        else:
            return img


