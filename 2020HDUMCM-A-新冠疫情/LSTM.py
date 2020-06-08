# Writer:bevarb
# Time: 2020/06/06

import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理
df = pd.read_csv('./Data/ChinaHubei.csv')
df.set_index(["Date"], inplace=True)
All_value = np.array(df.loc["1/22/20":"6/3/20", "Confirmed"])

N = 70000
TRAIN_SIZE = 0.2

x = []
y = []
seq = 3  # 选择几天的数据作为小输入
feat_nums = 1
for i in range(len(All_value)-seq-1):
    x.append(All_value[i:i+seq])
    y.append(All_value[i+seq])

Data_Length = len(y)
flag1 = int(TRAIN_SIZE * Data_Length)   # 训练集与测试集的分界线


train_x = (torch.tensor(x[0:flag1]).float()/N).reshape(-1, seq, feat_nums)
train_y = (torch.tensor(y[0:flag1]).float()/N).reshape(-1, 1)
test_x = (torch.tensor(x[flag1:Data_Length]).float()/N).reshape(-1, seq, feat_nums)
test_y = (torch.tensor(y[flag1:Data_Length]).float()/N).reshape(-1, 1)
# print(test_y)

# 模型构建
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        self.linear = nn.Linear(16 * seq, 1)
    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 16 * seq)
        x = self.linear(x)
        return x

# 模型训练
model = LSTM()
optimzer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_func = nn.MSELoss()
model.train()

for epoch in range(1760):
    output = model(train_x)
    loss = loss_func(output, train_y)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    if epoch % 20 == 0:
        tess_loss = loss_func(model(test_x), test_y)
        print("epoch:{}, train_loss:{}, test_loss:{}".format(epoch, loss, tess_loss))

# 模型预测、画图
model.eval()
prediction = list((model(train_x).data.reshape(-1))*N) + list((model(test_x).data.reshape(-1))*N)
plt.plot(y, label='True Value')
plt.plot(prediction[:flag1], label='LSTM fit')
plt.plot(np.arange(flag1, Data_Length, 1), prediction[flag1:Data_Length], label='LSTM pred')
# print(len(All_value[3:]))
# print(len(prediction[40:]))
plt.legend(loc='best')
plt.title('Cumulative infections prediction(USA)')
plt.xlabel('Day')
plt.ylabel('Cumulative Cases')
plt.savefig("./Images/LSTM.tif")
plt.close()
