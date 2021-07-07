import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import Unet

"""
width_out : width of the output image
height_out : height of the output image
width_in : width of the input image
height_in : height of the input image
"""

unet = Unet.Unet(inchannels, outchannnels)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(unet.parameters(), lr = 0.01, momentum=0.99)
outputs = outputs.permute(0, 2, 3, 1)
m = outputs.shape[0]
outputs = outputs.resize(m*width_out*height_out, 2)
labels = labels.resize(m*width_out*height_out)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()