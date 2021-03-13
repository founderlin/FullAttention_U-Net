import os
import cv2
import glob

from torch.optim import optimizer
from torch.utils.data import Dataset
import random
import torch
from torch import optim, tensor
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


class DataLoader(Dataset):
    def __init__(self, data_path):
        # read images for training
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))

    def augment(self, image, flipCode):
        # data enrichment using cv2.flip
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # generate path of each image via index
        image_path = self.imgs_path[index]        

        # generate path of label
        label_path = image_path.replace('image', 'label')

        # read all images and labels
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # convert RGB to one-channel (black and white)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        # process labels, switch 255 to 1
        if label.max() > 1:
            label = label / 255

        # data enrichment
        flipCode = random.choice([-1, 0, 1, 2])
        flipCode = random.choice([])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)

        return image, label

    def __len__(self):
        # get the size of data set
        return len(self.imgs_path)
