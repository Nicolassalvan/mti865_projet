from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import random, randint, uniform

from torchvision.transforms.v2 import functional as F

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
    # seed 
    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False, seed=42, method='method1'):
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
        np.random.seed(seed) # seed for reproducibility 
        self.method = method #getitem1 ou getitem2

    def __getitem__(self, index):
        if self.method == 'method1':
            return self.__getitem1__(index)
        elif self.method == 'method2':
            return self.__getitem2__(index)
        else:
            raise ValueError("Méthode de sélection invalide. Choisir 'method1' ou 'method2'")



    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        # Use pyTorch functional transforms to augment image and mask at the same time

        if np.random.rand() > 0.5:
            img = F.horizontal_flip(img)
            mask = F.horizontal_flip(mask) if mask is not None else None
        if np.random.rand() > 0.5:
            img = F.vertical_flip(img)
            mask = F.vertical_flip(mask) if mask is not None else None
        if np.random.rand() > 0.5:
            # angle = random() * 60 - 30
            angle = uniform(-5, 5)
            img = F.rotate(img, angle)
            mask = F.rotate(mask, angle) if mask is not None else None
        if np.random.rand() > 0.5:
            # x-axis translation between -5 and 5 pixels
            translate_x = uniform(-5, 5)
            img = F.affine(img, angle=0, translate=(translate_x, 0), scale=1, shear=0)
            mask = F.affine(mask, angle=0, translate=(translate_x, 0), scale=1, shear=0) if mask is not None else None
        if np.random.rand() > 0.5:
            # y-axis translation between -5 and 5 pixels
            translate_y = uniform(-5, 5)
            img = F.affine(img, angle=0, translate=(0, translate_y), scale=1, shear=0)
            mask = F.affine(mask, angle=0, translate=(0, translate_y), scale=1, shear=0) if mask is not None else None
        # if random() > 0.5:
        #     # Rescale between 0.6 and 1.4
        #     img = F.affine(img, angle=0, translate=(0, 0), scale=(0.6, 1.4), shear=0)
        #     mask = F.affine(mask, angle=0, translate=(0, 0), scale=(0.6, 1.4), shear=0)
        return img, mask

    def __getitem1__(self, index):

        if self.mode in ['train', 'val', 'test']:
            img_path, mask_path = self.imgs[index]
            img = Image.open(img_path)
            mask = Image.open(mask_path).convert('L')
        else:
            img_path, mask_path = self.imgs[index]
            img = Image.open(img_path)
            mask = None # pas de masque pour le moment 

        if self.equalize:
            img = F.equalize(img)

        if self.augmentation :
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            if mask is not None:
                mask = self.mask_transform(mask)

        return [img, mask, img_path] if self.mode in ['train', 'val', 'test'] else [img, None, img_path]

    def __getitem2__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path)
        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')
        else:
            mask = None

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation and mask is not None:
            img, mask = self.augment(img, mask)


        if self.transform:
            img = self.transform(img)
            if mask_path is not None:
                mask = self.mask_transform(mask)
        
        if mask_path is not None:
            return [img, mask, img_path]
        else:
            return [img, img_path]

