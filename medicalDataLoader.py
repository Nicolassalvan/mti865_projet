from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

# Ignore warnings
import warnings

import pdb

warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train', 'train-unlabelled', 'val', 'test']
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


    elif mode == 'train-unlabelled' :
        train_img_path = os.path.join(root, 'train', 'Img-Unlabeled')
        unlabeled_images = os.listdir(train_img_path)
        unlabeled_images.sort()

        for it_im in unlabeled_images:
            item = (os.path.join(train_img_path, it_im), None)  # pas de masque pour le moment
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
    print('Found %d items in %s' % (len(items), mode))
    print('First item: ', items[0])
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
        if random() > 0.5:
            img = ImageOps.flip(img)
            if mask is not None:
                mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            if mask is not None:
                mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 60 - 30
            img = img.rotate(angle)
            if mask is not None : 
                mask = mask.rotate(angle)
        return img, mask

    def __getitem__(self, index):

        if self.mode in ['train', 'val', 'test']:
            img_path, mask_path = self.imgs[index]
            img = Image.open(img_path)
            mask = Image.open(mask_path).convert('L')
        else:
            img_path, mask_path = self.imgs[index]
            img = Image.open(img_path)
            mask = None # pas de masque pour le moment 

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            if mask is not None:
                mask = self.mask_transform(mask)

        return [img, mask, img_path] if self.mode in ['train', 'val', 'test'] else [img, None, img_path]