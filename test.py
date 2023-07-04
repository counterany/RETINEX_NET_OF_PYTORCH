from model import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from utils import *


def load_image(img):
    img = np.array(img, dtype='float32') / 255.0
    # 归一化
    img = np.float32((img - np.min(img)) / np.maximum((np.max(img) - np.min(img)), 0.001))
    return img


def read_load_image2(file_path):
    """
    获得所有图片的矩阵和形状
    :param file_path: 文件夹位置
    :return: 图片名称的集合
    """
    all_img = []
    shape_m = []
    for filename in os.listdir(file_path):  # 返回指定的文件夹包含的文件或文件夹的名字的列表。
        img = cv2.imread(file_path + "/" + filename)
        shape_m.append(img.shape)
        img = cv2.resize(img, (512, 512))  # 将所有图片压缩为（512， 512）
        img = load_image(img)
        all_img.append(img)
    return all_img, shape_m
    

save_dir = "G:\\my_baby\\retinex\\decom_train\\last"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 准备测试数据
eval_image, eval_shape = read_load_image2("G:\\junior\\junior_next_term\\professional  introduction"
                                          "\\my_kind\\venv\\LOLdataset\\eval15\\low")
eval_len = len(eval_image)
# 加载模型
d_CKPT_PATH = "MyNet_Decom1000_best.pkl"
d_checkpoints = torch.load(d_CKPT_PATH)
d_checkpoint = d_checkpoints['state_dict']
d_model = decom_net().cuda()
d_model.load_state_dict(d_checkpoint)
d_model.cpu()
d_model.eval()

a_CKPT_PATH = "MyNet_adjust110_best.pkl"
a_checkpoints = torch.load(a_CKPT_PATH)
a_checkpoint = a_checkpoints['state_dict']
a_model = adjust_net().cuda()
a_model.load_state_dict(a_checkpoint)
a_model.cpu()
a_model.eval()

for i in range(0, eval_len):
    eval_image1 = eval_image[i]
    eval_image1 = torch.tensor(eval_image1)
    eval_image1 = eval_image1.reshape(1, eval_image1.shape[0], eval_image1.shape[1], eval_image1.shape[2])  # 网络输入为四维
    eval_image1 = eval_image1.permute([0, 3, 1, 2])
    eval_R, eval_I = d_model(eval_image1)
    eval_II = a_model(eval_I, eval_R)
    eval_II = torch.cat((eval_II, eval_II, eval_II), dim=1)
    image = eval_R * eval_II
    # 对增强后图片的处理
    image = image.squeeze(0)
    image = image.permute([1, 2, 0])
    image = image.detach().numpy()
    one_shape = eval_shape[i]
    new_shape = (int(one_shape[1]), int(one_shape[0]))  # 照片的原始维度（长和宽）
    image = cv2.resize(image, new_shape)  # 将照片还原成原始维度
    # 对原始图片的处理
    eval_image1 = eval_image1.squeeze(0)
    eval_image1 = eval_image1.permute([1, 2, 0])
    eval_image1 = eval_image1.detach().numpy()
    eval_image1 = cv2.resize(eval_image1, new_shape)
    # 保存两张图片
    save_image(os.path.join(".\\adjust_train\\retinex", f"__{i}.png"), eval_image1, image)
    # 对分解网络获得的反射图和光照图进行处理
    # 对光照图进行处理
    eval_I = torch.cat((eval_I, eval_I, eval_I), dim=1)
    eval_I = eval_I.squeeze(0)
    eval_I = eval_I.permute([1, 2, 0])
    eval_I = eval_I.detach().numpy()
    eval_I = cv2.resize(eval_I, new_shape)
    # 对反射率进行处理
    eval_R = eval_R.squeeze(0)
    eval_R = eval_R.permute([1, 2, 0])
    eval_R = eval_R.detach().numpy()
    eval_R = cv2.resize(eval_R, new_shape)
    # 保存图片
    save_image(os.path.join(save_dir, f"__{i}.png"), eval_R, image)
