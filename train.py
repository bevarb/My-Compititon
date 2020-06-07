
import argparse
import torch
from torch.optim import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
from Dataloader import get_train_test
from RNNmodel import SEIR_RNN

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=50, help="epoch to start training from")

opt = parser.parse_args()

mse_loss = torch.nn.MSELoss(reduction='none')


def ode_loss(y_true, y_pred):
    mask = torch.sum(y_true, dim=-1, keepdim=True) > 0
    mask = mask.float()

    return torch.sum(mask * mse_loss(y_true, y_pred)) / mask.sum()




steps, h = 50, 1
nums = 3.282 * (10 ** 9)
N = 5000000

x, y, train_x, train_y, test_x, test_y, Data_Length, flag1 = get_train_test('./Data/US_All.csv', "1/22/20", "5/15/20", nums, N)

model = SEIR_RNN(steps, h, nums)
optimizer = Adam(model.parameters(), lr=1e-2)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                     mode='min',  # 检测指标
                                     factor=0.3,  # 学习率调整倍数
                                     patience=5,  # 忍受多少个Epoch无变化
                                     verbose=True,  # 是否打印学习率信息
                                     # threshold=0.0001,
                                     # threshold_mode='rel',
                                     cooldown=3,  # 冷却时间
                                     min_lr=0,   # 学习率下限
                                     eps=1e-07   # 学习率衰减的最小值
                                    )
loss_min = 1000000




All_loss = []
for epoch in range(opt.epoch):
    outputs = model(train_x)
    loss = ode_loss(train_y, outputs)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(epoch, loss.item())
    All_loss.append(loss.item())
    scheduler.step(loss_min)
    if loss_min > loss.item():
        loss_min = loss.item()

    if epoch == opt.epoch - 1:
        torch.save(model.state_dict(), "SEIR.pth")
    



# plt.plot(np.arange(0, flag1, 1), All_loss, label='Train_Loss')
# # plt.plot(np.arange(flag1, Data_Length, 1), prediction_test[:, 0, 0, 2], label='SEIR pred')
#
# plt.legend(loc='best')
# plt.title('Cumulative infections prediction(USA)')
# plt.xlabel('Day')
# plt.ylabel('Cumulative Cases')
# plt.savefig('./Images/1.png')
# plt.close()



