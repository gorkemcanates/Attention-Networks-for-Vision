__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import transforms as Ap
from torch.utils.data import Dataset
from PIL import Image
import io
import matplotlib.pyplot as plt
import tarfile



class ImageNet:
    def __init__(self,
                 train_path,
                 val_path,
                 train_label_path,
                 val_label_path,
                 blacklist=None,
                 train_transform=None,
                 val_transform=None,

                 ):
        self.train_path = train_path
        self.val_path = val_path
        self.train_label_path = train_label_path
        self.val_label_path = val_label_path
        self.blacklist = blacklist
        self.train_transform = train_transform
        self.val_transform = val_transform


    def read_train_data(self, impath, lpath):

        if not os.path.exists(impath):
            os.mkdir(impath)
        if not os.path.exists(lpath):
            os.mkdir(lpath)

        with open(self.train_label_path) as f:
            train_label_info = f.readlines()
        train_label_sp = []
        for file in train_label_info:
            train_label_sp.append(file[:-1].split(' '))
        count = 1
        labeltxt = open(lpath + 'train_labels.txt', 'w+')
        for f in train_label_sp:
            label, name = f[1], f[2]
            tar = tarfile.open(self.train_path + f[0] + '.tar')
            for m in tar.getmembers():
                if count >= 396399:
                    f = tar.extractfile(m)
                    f = f.read()
                    image = Image.open(io.BytesIO(f))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(impath + str(count) + '.jpeg')
                labeltxt.write(str(label) + '\n')
                count +=1
        labeltxt.close()


    def rename(self):
        f = os.listdir(self.val_path)
        count = 1
        for fname in f:
            os.rename(self.val_path + fname, self.val_path + str(count) + '.JPEG')
            count += 1

class CIFAR:
    def __init__(self,
                 train_path,
                 val_path,
                 train_label_path,
                 val_label_path,
                 case='100',
                 ):
        self.train_path = train_path
        self.val_path = val_path
        self.train_label_path = train_label_path
        self.val_label_path = val_label_path



    def read_train_data(self, impath, lpath):

        if not os.path.exists(impath):
            os.mkdir(impath)
        if not os.path.exists(lpath):
            os.mkdir(lpath)

        with open(self.train_label_path) as f:
            train_label_info = f.readlines()
        train_label_sp = []
        for file in train_label_info:
            train_label_sp.append(file[:-1].split(' '))
        count = 1
        labeltxt = open(lpath + 'train_labels.txt', 'w+')
        for f in train_label_sp:
            label, name = f[1], f[2]
            tar = tarfile.open(self.train_path + f[0] + '.tar')
            for m in tar.getmembers():
                if count >= 396399:
                    f = tar.extractfile(m)
                    f = f.read()
                    image = Image.open(io.BytesIO(f))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(impath + str(count) + '.jpeg')
                labeltxt.write(str(label) + '\n')
                count +=1
        labeltxt.close()


    def rename(self):
        f = os.listdir(self.val_path)
        count = 1
        for fname in f:
            os.rename(self.val_path + fname, self.val_path + str(count) + '.JPEG')
            count += 1


if __name__ == '__main__':
    dataset = CIFAR(train_path='C:\GorkemCanAtes\PycharmProjects\SOTA/train_raw/',
                      val_path='C:\GorkemCanAtes\PycharmProjects\SOTA\ImageNet\Dataset/validation/',
                      train_label_path='C:\GorkemCanAtes\PycharmProjects\SOTA\ImageNet\Dataset\labels_raw/train_ground_truth_raw.txt',
                      val_label_path='C:\GorkemCanAtes\PycharmProjects\SOTA\ImageNet\Dataset\labels_raw/validation_ground_truth.txt',
                      )
    # dataset.read_train_data(impath='C:\GorkemCanAtes\PycharmProjects\SOTA\ImageNet\Dataset/train/',
    #                         lpath='C:\GorkemCanAtes\PycharmProjects\SOTA\ImageNet\Dataset\labels/')



