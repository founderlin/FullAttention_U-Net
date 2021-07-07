import os
import torch
import cv2
import glob
import numpy as np
from torch import optim, nn
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torchvision import transforms
import random
from unet_model import UNet


def augment(image, flipCode):
    # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
    flip = cv2.flip(image, flipCode)
    return flip

data_path = "data/train10/"
imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
# print(len(imgs_path))
for i in imgs_path:
    print(i)
    # image_path = imgs_path[1]
    image_path = i
    print(image_path)
    label_path = image_path.replace('image', 'label')
    print(label_path)
    image = cv2.imread(image_path)
    label = cv2.imread(label_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    print(np.shape(image))

    print(label.max())

    # if label.max() > 1:
    #     label = label / 255
    #
    # print(label.max())

    # flipCode = random.choice([-1, 0, 1, 2])
    # if flipCode != 2:
    #     image = augment(image, flipCode)
    #     label = augment(label, flipCode)

    plt.imshow(image)
    plt.show()
    plt.imshow(label)
    plt.show()
# image = image.reshape(1, image.shape[0], image.shape[1])
# label = label.reshape(1, label.shape[0], label.shape[1])
#
# image = torch.from_numpy(image)
# label = torch.Tensor(label)
#
# image = transforms.functional.to_pil_image(image)
# label = transforms.functional.to_pil_image(label)
#
# image = transforms.functional.resize(image, 256)
# label = transforms.functional.resize(label, 256)
#
# print(np.shape(image))
# plt.imshow(image)
# plt.show()
#
# plt.imshow(label)
# plt.show()
#
# image = transforms.functional.to_tensor(image)
# label = transforms.functional.to_tensor(label)

# image = torch.utils.data.DataLoader(
#     dataset=image, batch_size=1, shuffle=True
#     )
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# image = image.to(device=device, dtype=torch.float32)
# # label = label.to(device=device, dtype=torch.float32)
#
#
# print(image.shape)
# net = UNet(n_channels=1, n_classes=1)
# net.to(device=device)
# net(image)


