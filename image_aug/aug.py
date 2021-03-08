import cv2
import torch
import torchvision
import numpy as np
import PIL.ImageOps
from torch import nn
from PIL import Image

image =Image.open('grib.jpg')
image = PIL.ImageOps.invert(image)
data = np.asarray(image)
# window_name = 'image'
# image.show()

aug = torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5)


def apply(image, aug, num_rows=2, num_cols=4):
    new = [aug(image) for _ in range(num_rows * num_cols)]
    new = [torchvision.transforms.functional.adjust_hue(im, -0.5) for im in new]
    print(new[0])
    all_new = np.concatenate(new, axis=1)
    print(all_new)
    all_new_pil = Image.fromarray(all_new)
    all_new_pil.show()
    all_new_pil.save('grib2.jpg') 

apply(image, aug)   