__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import os
from PIL import Image
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelBinarizer


class ImageNetDataset:
    def __init__(self,
                 train_image_dir,
                 val_image_dir,
                 train_label_dir,
                 val_label_dir,
                 train_transforms,
                 val_transforms,
                 debug=False):

        self.train_dataset = ImageNet(image_dir=train_image_dir,
                                      label_dir=train_label_dir,
                                      transform=train_transforms,
                                      debug=debug)

        self.test_dataset = ImageNet(image_dir=val_image_dir,
                                      label_dir=val_label_dir,
                                      transform=val_transforms,
                                     debug=debug)


class CIFARDataset:
    def __init__(self,
                 path,
                 train_transform,
                 test_transform,
                 dataset='CIFAR100/'):
        label_binarizer = LabelBinarizer()
        if dataset == 'CIFAR10/':
            self.train_dataset = torchvision.datasets.CIFAR10(root=path,
                                                              train=True,
                                                              transform=train_transform,
                                                              download=True)
            self.test_dataset = torchvision.datasets.CIFAR10(root=path,
                                                             train=False,
                                                             transform=test_transform,
                                                             download=True)
        elif dataset == 'CIFAR100/':
            self.train_dataset = torchvision.datasets.CIFAR100(root=path,
                                                              train=True,
                                                              transform=train_transform,
                                                               download=True)
            self.test_dataset = torchvision.datasets.CIFAR100(root=path,
                                                             train=False,
                                                             transform=test_transform,
                                                              download=True)
        else:
            raise AttributeError

        self.train_dataset.targets = np.expand_dims(label_binarizer.fit_transform(self.train_dataset.targets), axis=2)
        self.test_dataset.targets = np.expand_dims(label_binarizer.fit_transform(self.test_dataset.targets), axis=2)


class ImageNet(Dataset):
    def __init__(self,
                 image_dir,
                 label_dir,
                 transform=None,
                 debug = False):
        self.image_dir = image_dir
        self.transforms = transform
        self.images = []
        l =  os.listdir(image_dir).__len__()
        if debug:
            self.images = [str(i+1) + '.jpeg' for i in range(256)]

            with open(os.path.join(label_dir, 'labels.txt')) as f:
                labels = f.readlines()
                labels = labels[:256]
        else:
            self.images = [str(i+1) + '.jpeg' for i in range(l)]
            with open(os.path.join(label_dir, 'labels.txt')) as f:
                labels = f.readlines()
        f.close()

        self.labels = []
        for value in labels:
            self.labels.append(int(value[:-1]))

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.images[item])
        image = Image.open(img_path).convert("RGB")
        label_init = self.labels[item]
        label = torch.zeros((1000, 1), dtype=torch.float32)
        label[label_init-1] = 1

        if self.transforms is not None:
            image = self.transforms(image)
        return image, label


    def __len__(self):
        return len(self.images)


