import os
import cv2
import numpy as np
import imghdr

'''read images from filepath'''


def read_ima(filepath):
    ima_list = []
    father_path = os.listdir(filepath)
    for allDir in father_path:
        child_path = os.path.join(filepath, allDir)
        ima_list.append(child_path)
    return ima_list


'''processing images, half size'''
def proce_ima(filepath,after_path):
    files = read_ima(filepath)
    i = 0
    for file in files:
        if imghdr.what(file) in ('bmp', 'jpg', 'png', 'jpeg'):
            img = cv2.imread(file)
            '''half cut'''
            # half_ima = img[0:720, 0:640]
            # # cv2.imwrite(file.split('Datasets')[0] + after_pth + file.split(filepath)[1], half_ima)
            # half_img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
            # cv2.imwrite(file.split('Datasets')[0]+after_path+file.split(filepath)[1],half_img)
            list_img = ['left_up','right_up','left_buttom','right_buttom','center','small']
            for place in list_img:
                if place == 'left_up':
                    crop_img = img[0:320,0:640]
                if place == 'right_up':
                    crop_img = img[0:320,640:1280]
                if place == 'left_buttom':
                    crop_img = img[400:720,0:640]
                if place == 'right_buttom':
                    crop_img = img[400:720,640:1280]
                if place == 'center':
                    crop_img = img[200:520,320:960]
                if place == 'small':
                    crop_img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
                    crop_img = crop_img[0:320,0:640]
            #
            #
                cv2.imwrite(file.split('Datasets')[0] + after_path + '/' + file.split(filepath+'/')[1].split('.jpg')[0]
                             + place + '.jpg', crop_img)
                # cv2.imwrite(file.split('Datasets')[0] + after_path + '/' + file.split(filepath + '/')[1]
                #             + place + '.jpg', crop_img)



filepath = '/home/dan/linda_project/linda_data/Datasets/Hibikino dataset/trainColor/color'
after_pth = 'Datasets/Hibikino dataset/five_small_col/col'
proce_ima(filepath,after_pth)

