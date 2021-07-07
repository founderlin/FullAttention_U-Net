
import os
import glob
import cv2

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

class ImgProcesser(Dataset):
    def __init__(self, data_path):
        # read images for training
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(self.data_path, 'train/image/*.png'))
        self.tests_path = glob.glob(os.path.join(self.data_path, 'test/*.JPG'))
        # print(self.imgs_path)

    def getTrainItem(self):
        count = 1
        for image_path in self.imgs_path:
            if count < 10:
                count_S = "00"+str(count)
            elif count > 99:
                count_S = str(count)
            else:
                count_S = "0"+str(count)

            image = Image.open(image_path).convert('LA')
            image = transforms.functional.resize(image, 512)
            image.save('data100X/train/image/' + count_S + '.png', mode='png')

            label_path = image_path.replace('image', 'label')
            label = Image.open(label_path).convert('LA')
            label = transforms.functional.resize(label, 512)
            label.save('data100X/train/label/' + count_S + '.png', mode='png')
            count += 1

    def getTestItem(self):
        count = 1
        for test_path in self.tests_path:
            if count < 10:
                count_S = "00"+str(count)
            elif count > 99:
                count_S = str(count)
            else:
                count_S = "0"+str(count)
            print(test_path)
            image = Image.open(test_path).convert('LA')
            image = transforms.functional.resize(image, 512)
            image.save('data100X/test/' + count_S + '.png', mode='png')

            count += 1


if __name__ == "__main__":
    dataset = ImgProcesser("data100/")
    dataset.getTrainItem()
    dataset.getTestItem()