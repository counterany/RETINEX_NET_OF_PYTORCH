import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def load_image(img):
    img = np.array(img, dtype='float32') / 255.0
    # 归一化
    img = np.float32((img - np.min(img)) / np.maximum((np.max(img) - np.min(img)), 0.001))
    return img


def read_load_image(file_path):
    all_img = []
    for file_name in os.listdir(file_path):
        img = cv2.imread(file_path + "/" + file_name)
        img = load_image(img)
        all_img.append(img)
    return all_img


def data_agument(img, mode):
    """
    将数据进行一定随机的转化
    :param img: 图片数据
    :param mode: 变化类型
    :return: 转换后的图片数据
    """
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)  # np.flipud()用于翻转列表，将矩阵进行上下翻转
    elif mode == 2:
        return np.rot90(img)  # #将矩阵img逆时针旋转90°
    elif mode == 3:
        return np.flipud(np.rot90(img))  # #旋转90°后在上下翻转
    elif mode == 4:
        return np.rot90(img, k=2)  # 将矩阵img逆时针旋转90°*k
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    else:
        return print("error")


def save_image(file_path, result_1, result_2=None):
    """
    保存训练照片
    :param file_path: 存放位置
    :param result_1: 反射图
    :param result_2: 光照图
    """
    # result_1 = np.squeeze(result_1)  # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉，即还原维度
    # result_2 = np.squeeze(result_2)
    image_cat = np.concatenate((result_1, result_2), axis=1)  # 将两张照片进行拼接
    cv2.imwrite(file_path, image_cat * 255.0)  # 将拼接后的照片写入文件中
    print(file_path)

def gradient(input_tensor, direction):
    kernel_x = [[0, 0], [-1, 1]]
    kernel_x = torch.FloatTensor(kernel_x).view((1, 1, 2, 2)).cuda()
    kernel_y = kernel_x.permute([0, 1, 3, 2])
    if direction == "x":
        kernel = kernel_x
    elif direction == "y":
        kernel = kernel_y
    # weight = nn.Parameter(data=kernel, requires_grad=False)
    gradient_orig = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
    return gradient_orig


def ave_gradient(input_tensor, direction):
    out = F.avg_pool2d(gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1)
    return out


def smooth(input_i, input_r):
    input_r = 0.299 * input_r[:, 0, :, :] + 0.587 * input_r[:, 1, :, :] + 0.114 * input_r[:, 2, :, :]
    input_r = torch.unsqueeze(input_r, dim=1)
    i_x_gradient = gradient(input_i, "x")
    i_y_gradient = gradient(input_i, "y")
    r_x_gradient = ave_gradient(input_r, "x")
    r_y_gradient = ave_gradient(input_r, "y")
    smooth_loss = torch.mean(
        i_x_gradient * torch.exp(-10 * r_x_gradient) + i_y_gradient * torch.exp(-10 * r_y_gradient))
    return smooth_loss