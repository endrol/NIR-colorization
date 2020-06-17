import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/dan/linda_project2/linda_data/Datasets/loveday/train-col/'
                 'col/scen5/scene5/Scene_5_00059_V.png', cv2.IMREAD_COLOR)

gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

b, g, r = cv2.split(img)


print('thios ')
# equr = cv2.equalizeHist(r)
# equg = cv2.equalizeHist(g)
# equb = cv2.equalizeHist(b)
# merged = cv2.merge([equr,equg,equb])
# plt.figure()
# plt.subplot(2,3,4)
# plt.imshow(cv2.merge([r,g,b]))
# plt.subplot(2,3,1)
# plt.imshow(gray_img, cmap='gray')
# equ = cv2.equalizeHist(gray_img)
# plt.subplot(2,3,2)
# plt.imshow(equ, cmap='gray')
# plt.subplot(2,3,5)
# plt.imshow(merged)
#
# clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
# cl1 = clahe.apply(gray_img)
# plt.subplot(2,3,3)
# plt.imshow(cl1, cmap='gray')
# clr = clahe.apply(r)
# clg = clahe.apply(g)
# clb = clahe.apply(b)
# merged2 = cv2.merge([clr, clg, clb])
# plt.subplot(2,3,6)
# plt.imshow(merged2)
#
# plt.show()
transforms.ToPILImage()(img_PIL_Tensor).convert('RGB')
