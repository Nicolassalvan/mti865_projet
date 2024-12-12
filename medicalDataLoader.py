from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import random, randint, uniform

from torchvision.transforms.v2 import functional as F

from torchvision.transforms.v2 import ElasticTransform, ColorJitter
import matplotlib.pyplot as plt

# Ignore warnings
import warnings

import pdb

warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train','val', 'test']
    items = []

    if mode == 'train':
        train_img_path = os.path.join(root, 'train', 'Img')
        train_mask_path = os.path.join(root, 'train', 'GT')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)


    elif mode == 'val':
        val_img_path = os.path.join(root, 'val', 'Img')
        val_mask_path = os.path.join(root, 'val', 'GT')

        images = os.listdir(val_img_path)
        labels = os.listdir(val_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(val_img_path, it_im), os.path.join(val_mask_path, it_gt))
            items.append(item)
    else:
        test_img_path = os.path.join(root, 'test', 'Img')
        test_mask_path = os.path.join(root, 'test', 'GT')

        images = os.listdir(test_img_path)
        labels = os.listdir(test_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(test_img_path, it_im), os.path.join(test_mask_path, it_gt))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        # Use pyTorch functional transforms to augment image and mask at the same time

        if random() > 0.5:
            img = F.horizontal_flip(img)
            mask = F.horizontal_flip(mask)
        if random() > 0.5:
            img = F.vertical_flip(img)
            mask = F.vertical_flip(mask)
        if random() > 0.5:
            # angle = random() * 60 - 30
            angle = uniform(-5, 5)
            img = F.rotate(img, angle)
            mask = F.rotate(mask, angle)
        if random() > 0.5:
            # x-axis translation between -5 and 5 pixels
            translate_x = uniform(-5, 5)
            img = F.affine(img, angle=0, translate=(translate_x, 0), scale=1, shear=0)
            mask = F.affine(mask, angle=0, translate=(translate_x, 0), scale=1, shear=0)
        if random() > 0.5:
            # y-axis translation between -5 and 5 pixels
            translate_y = uniform(-5, 5)
            img = F.affine(img, angle=0, translate=(0, translate_y), scale=1, shear=0)
            mask = F.affine(mask, angle=0, translate=(0, translate_y), scale=1, shear=0)
        # if random() > 0.5:
        #     elastic_transformer = ElasticTransform(alpha=250.0)
        #     img = elastic_transformer(img)
        #     mask = elastic_transformer(mask)
        #     print("elastictrans")
            #print('Image batch dimensions: ', img.size())

            #plot([img]+mask)

        # if random() > 0.5:
        #     jitter = ColorJitter(brightness=.5, hue=.3)
        #     img = jitter(img)
        #     mask = mask
            # print("jitter")
            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(img, cmap='gray')
            # plt.title('Image')
            # plt.axis('off')
            # plt.subplot(1,2,2)
            # plt.imshow(mask, cmap='gray')
            # plt.title('Mask')
            # plt.axis('off')
            # plt.show()
        # if random() > 0.5:
        #     # Rescale between 0.6 and 1.4
        #     img = F.affine(img, angle=0, translate=(0, 0), scale=(0.6, 1.4), shear=0)
        #     mask = F.affine(mask, angle=0, translate=(0, 0), scale=(0.6, 1.4), shear=0)
        return img, mask

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path)
        mask = Image.open(mask_path).convert('L')

        if self.equalize:
            img = F.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)

        return [img, mask, img_path]