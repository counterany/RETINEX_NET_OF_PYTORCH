from model import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from utils import *

epoch = 120  
batch_size = 16
patch_size = 96
learning_rate = 0.001
save_dir = "./adjust_train"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


class deal_data(Dataset):
    def __init__(self):
        self.low_image = read_load_image("G:\\junior\\junior_next_term\\professional  introduction\\my_kind"
                                         "\\venv\\LOLdataset\\our485\\low")
        self.high_image = read_load_image("G:\\junior\\junior_next_term\\professional  introduction\\my_kind"
                                          "\\venv\\LOLdataset\\our485\\high")
        self.len = len(self.low_image)

    def __getitem__(self, item):
        high, width, _ = self.low_image[item].shape
        x = random.randint(0, high - patch_size)
        y = random.randint(0, width - patch_size)
        mode = random.randint(0, 7)
        low = data_agument(self.low_image[item][x:x + patch_size, y:y + patch_size, :], mode)
        high = data_agument(self.high_image[item][x:x + patch_size, y:y + patch_size, :], mode)
        low = low.copy()
        high = high.copy()
        low = torch.tensor(low)
        high = torch.tensor(high)
        return low, high

    def __len__(self):
        return self.len


def adjust_grad_loss(input_i_low, input_i_high):
    """
    调整网络的梯度损失函数
    :param input_i_low: 输入的目标图
    :param input_i_high: 输入的光照图
    :return: 损失值
    """
    x_one = torch.sub(gradient(input_i_low, 'x'), gradient(input_i_high, 'x'))
    y_one = torch.sub(gradient(input_i_low, 'y'), gradient(input_i_high, 'y'))  
    x_loss = torch.pow(x_one, 2)
    y_loss = torch.pow(y_one, 2)  
    all_grad_loss = torch.mean(x_loss + y_loss)  
    return all_grad_loss


class adjust_loss(nn.Module):
    def __init__(self):
        super(adjust_loss, self).__init__()

    def forward(self, R_low_data=None, high_img=None, I_adjust=None, input_high_i=None):
        I_adjust_3 = torch.cat((I_adjust, I_adjust, I_adjust), dim=1)
        # 重构损失
        L_rec = torch.mean(torch.abs(R_low_data*I_adjust_3 - high_img))
        # 光照图的光滑
        L_smooth = smooth(I_adjust, R_low_data)
        # 目标图像和输入的光照图像各自的梯度绝对值相减后的L2范数作为loss的组成部分
        grad_loss = adjust_grad_loss(I_adjust, input_high_i)
        # 目标值 - 输入的光照图的L2范数作为loss的组成部分
        sub_loss = torch.sub(I_adjust, input_high_i)
        L2_loss = torch.mean(torch.pow(sub_loss, 2))
        rec_alpha = torch.tensor([1]).cuda()
        is_alpha = torch.tensor([3]).cuda()
        adjust_Loss = torch.mul(rec_alpha, L_rec) + torch.mul(is_alpha, L_smooth) + grad_loss + L2_loss
        return adjust_Loss


# 准备数据
deal_dataset = deal_data()
train_data = DataLoader(dataset=deal_dataset, batch_size=batch_size, shuffle=True)
# 加载训练好了的分解网络模型
d_CKPT_PATH = f"MyNet_Decom1000_best.pkl"
d_checkpoints = torch.load(d_CKPT_PATH)
d_checkpoint = d_checkpoints['state_dict']
d_model = decom_net().cuda()
d_model.load_state_dict(d_checkpoint)
d_model.cpu()
d_model.eval()
# 准备模型
adjust_model = adjust_net().cuda()
# 配置优化器
adjust_optimizer = optim.Adam(adjust_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
# 配置损失函数
adjust_l = adjust_loss()
# 开始训练
for e in range(epoch):
    adjust_model.train()
    print(f"epoch:{e}")
    if e >= 21:
        learning_rate = learning_rate / 10.0
    for param_group in adjust_optimizer.param_groups:
        param_group['lr'] = learning_rate
    for i, data in enumerate(train_data):
        train_low, train_high = data
        train_low = train_low.permute([0, 3, 1, 2])
        train_high = train_high.permute([0, 3, 1, 2])
        # 把数据输入网络
        R_low, I_low = d_model(train_low)
        R_low = R_low.cuda()
        I_low = I_low.cuda()
        R_high, I_high = d_model(train_high)
        I_high = I_high.cuda()
        I_adjust = adjust_model(R_low, I_low)
        # I_adjust_3 = torch.cat((I_adjust, I_adjust, I_adjust), dim=1)
        train_high = train_high.cuda()
        loss = adjust_l(R_low, train_high, I_adjust, I_high).cuda()
        adjust_optimizer.zero_grad()
        loss.backward()
        adjust_optimizer.step()
        print("loss:", loss)
torch.save({'state_dict': adjust_model.state_dict(), 'epoch': epoch}, 'MyNet_adjust' + str(epoch) + '_best.pkl')
print('已经保存最优的模型参数')
