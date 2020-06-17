import os
import cv2
import numpy as np
import imghdr
from __future__ import unicode_literals, division
import numpy as np
from skimage import io, color
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from torch import nn
from torchvision import transforms, datasets
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import ImageFilter
import math

'''
this part tries to ignore the halation part when training and testing
'''

def read_ima(filepath):
    ima_list = []
    father_path = os.listdir(filepath)
    for allDir in father_path:
        child_path = os.path.join(filepath, allDir)
        ima_list.append(child_path)
    return ima_list

'''processing images to make the images' halation parts black/ white '''
def process_img(filepath, afterpath):
    files = read_ima(filepath)
    i = 0
    for file in files:
        if imghdr.what(file) in ('bmp', 'jpg', 'png', 'jpeg'):
            img = cv2.imread(file)
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            b,g,r = cv2.split(img)
