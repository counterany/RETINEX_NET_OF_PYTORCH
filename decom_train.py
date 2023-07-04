import torch

from model import *
from utils import *

epoch = 1000  # 迭代次数
batch_size = 16
patch_size = 96
learning_rate = 0.001
save_dir = "./decom_train"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


class rgb_gray(torch.nn.Module):
    def __init__(self):
        super(rgb_gray, self).__init__()
        kernel = [0.299, 0.587, 0.114]
        self.weight = torch.tensor(kernel).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        gray = F.conv2d(x, self.weight)
        return gray


def mutual_consistency_loss(input_I_low, input_I_mutual):
    """
    计算相互一致性
    :param input_I_low: 输入的低光照光照图
    :param input_I_mutual: 输入低光照图片
    :return:
    """
    gray = rgb_gray().cuda()  # 将低光照图转化为灰度图
    input_gray = gray(input_I_mutual)
    low_gradient_x = gradient(input_I_low, "x")
    gray_gradient_x = gradient(input_gray, "x")  # 获得光照图的梯度
    b = torch.Tensor([0.01]).cuda()
    x_loss = torch.abs(torch.div(low_gradient_x, torch.max(gray_gradient_x, b)))  # 利用公式写的程序
    low_gradient_y = gradient(input_I_low, "y")
    gray_gradient_y = gradient(input_gray, "y")
    y_loss = torch.abs(torch.div(low_gradient_y, torch.max(gray_gradient_y, b)))
    mutual_loss = torch.mean(x_loss + y_loss)  # 获得所有数据的平均值
    return mutual_loss


def mutual_i_loss(input_I_low, input_I_high):
    """
    计算Lld_is损失函数，平滑光照图
    :param input_I_low: 低光照图像的光照图
    :param input_I_high: 高光照图像的光照图
    :return: 平滑损失值
    """
    low_gradient_x = gradient(input_I_low, "x")
    high_gradient_x = gradient(input_I_high, "x")  # 计算x方向的梯度
    x_loss = torch.mul((low_gradient_x + high_gradient_x), torch.exp(-10 * (low_gradient_x + high_gradient_x)))
    low_gradient_y = gradient(input_I_low, "y")
    high_gradient_y = gradient(input_I_high, "y")  # 计算y方向的梯度
    y_loss = torch.mul((low_gradient_y + high_gradient_y),
                       torch.exp(-10 * (low_gradient_y + high_gradient_y)))
    mutual_loss = torch.mean(x_loss + y_loss)
    return mutual_loss


"""读取处理数据"""

#  为什么要对数据进行这样的处理？
# 这样处理可以减少网络的计算量（截取一小块）；可以增加样本的多样性和随机性（对数据进行随机翻转处理）
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

# 为什么这样构建损失函数？
# 1）重构损失：因为分解后的反射图和光照图重新构建出的照片应该跟原照片一样
# 2）反射一致性：因为相同的物体本身的颜色啊，形状啊什么的应该是一定的，所以高光照图像和低光照图像的反射图应该是一样的
# 3）光照的平滑性；就是光照图应该是分段平滑的
# 4）光照图应该跟输入图像去除颜色后的灰度图结构啥的相互一致
class decom_loss(nn.Module):
    def __init__(self):
        super(decom_loss, self).__init__()

    def forward(self, low_img, high_img, R_low, R_high, I_low_data, I_high_data):
        I_low_3 = torch.cat([I_low_data, I_low_data, I_low_data], dim=1)
        I_high_3 = torch.cat([I_high_data, I_high_data, I_high_data], dim=1)
        # L = L_recon + βL_ir + β1L_is
        # 重构损失
        L_recon = torch.mean(torch.abs(low_img - I_low_3 * R_low)) + torch.mean(torch.abs(high_img - I_high_3 * R_high))
        # 反射一致性
        L_ir = torch.mean(torch.abs(R_low - R_high))
        L1 = torch.mean(torch.abs(R_high * I_low_3 - low_img)) + torch.mean(torch.abs(R_low * I_high_3 - high_img))
        # 光照图的平滑，这里用的约束为两个光照图各自的梯度去除以它们梯度最大值后取绝对值相加
        I_mutual_loss = mutual_i_loss(I_low_data, I_high_data)  # 光照图的平滑损失值
        # 4） 相互一致性，即光照图和输入的一致性
        I_input_mutual_loss_low = mutual_consistency_loss(I_low_data, low_img)
        I_input_mutual_loss_high = mutual_consistency_loss(I_high_data, high_img)
        mc_LD_loss = I_input_mutual_loss_low + I_input_mutual_loss_high  # 光照图和原图的相互一致性
        # 系数
        rec_alpha = torch.tensor([1]).cuda()
        ir_alpha = torch.tensor([0.01]).cuda()
        is_alpha = torch.tensor([0.1]).cuda()
        L1_alpha = torch.tensor([0.001]).cuda()
        LD_alpha = torch.tensor([0.1]).cuda()
        Decom_loss = torch.mul(rec_alpha, L_recon) + torch.mul(ir_alpha, L_ir) \
                     + torch.mul(is_alpha, I_mutual_loss) + torch.mul(L1_alpha, L1) + torch.mul(LD_alpha, mc_LD_loss)
        return Decom_loss


# 准备数据
deal_dataset = deal_data()
train_data = DataLoader(dataset=deal_dataset, batch_size=batch_size, shuffle=True)
# 准备模型
decom_model = decom_net().cuda()
# 配置优化器
decom_optimizer = optim.Adam(decom_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
# 配置损失函数
decom_l = decom_loss()
# 开始训练
for e in range(epoch):
    decom_model.train()
    print(f"epoch:{e}")
    if e >= 21:
        learning_rate = learning_rate/10.0
    for param_group in decom_optimizer.param_groups:
        param_group['lr'] = learning_rate
    for i, data in enumerate(train_data):
        train_low, train_high = data
        # 为什么做这样的维度转换？
        # 因为cv2读取的时候，通道channel是放在最后的，但是网络要求通道放在第二
        train_low = train_low.permute([0, 3, 1, 2]).cuda()
        train_high = train_high.permute([0, 3, 1, 2]).cuda()
        # 把数据输入网络
        R_low, I_low = decom_model(train_low)
        R_high, I_high = decom_model(train_high)
        loss = decom_l(train_low, train_high, R_low, R_high, I_low, I_high).cuda()
        decom_optimizer.zero_grad()
        loss.backward()
        decom_optimizer.step()
        print("loss:", loss)
torch.save({'state_dict': decom_model.state_dict(), 'epoch': epoch}, 'MyNet_Decom' + str(epoch) + '_best.pkl')
print('已经保存最优的模型参数')
