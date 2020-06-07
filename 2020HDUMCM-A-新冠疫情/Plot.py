import torch
import numpy as np
from RNNmodel import SEIR_RNN
import matplotlib.pyplot as plt
from Dataloader import get_train_test
steps, h = 50, 1
nums = 3.282 * (10 ** 9)
N = 5000000

x, y, train_x, train_y, test_x, test_y, Data_Length, flag1 = get_train_test('./Data/UK.csv', "1/22/20", "5/15/20", nums, N)

model = SEIR_RNN(steps, h, N)
pre = torch.load("SEIR.pth")
model.load_state_dict(pre)

model.eval()
train_output = model(train_x).data



test_output = []
lii = torch.zeros(1, 1, 4)
lii[0, 0, :] = train_output[flag1 - 1, 0, 0, :]
test_output.append(train_output[flag1 - 1, 0, 0, :])
predict_length = 30  # 预测长度
for i in range(flag1, flag1 + predict_length):
    output__ = model(lii).data

    next_input = output__[0, 0, 0, :]
    lii[0, 0, :] = next_input
    test_output.append(output__[-1, 0, 0, :])

test_output = [i.numpy() for i in test_output]

test_output = np.squeeze(np.array(test_output)) * N


prediction_train = np.array(train_output * N)
# 模型预测、画图
 # + list((model(test_x).data) * N)
y = np.array(y)
#  plt.plot(np.arange(0, Data_Length, 1), y[:, 0], label='True Susceptible')
#     plt.plot(np.arange(0, Data_Length, 1), y[:, 1], label='True Exposed')
plt.plot(np.arange(0, Data_Length, 1), y[:, 2], label='True Infected')
# plt.plot(np.arange(0, Data_Length, 1), y[:, 3], label='True Removed')
# print(prediction_train.shape)
# print(prediction_train[:flag1, 0, 0, 0])
import pandas as pd
df = pd.read_csv('./Data/US_All.csv')
df.set_index(["Date"], inplace=True)
print(df.index)
plt.plot(np.arange(0, flag1, 1), prediction_train[:flag1, 0, 0, 2], label='SEIR fit')
# plt.plot(np.arange(flag1, Data_Length, 1), prediction_test[:, 0, 0, 2], label='SEIR pred')
plt.plot(np.arange(flag1, flag1 + predict_length + 1, 1), test_output[:, 2], label='SEIR pred')
plt.legend(loc='best')
plt.title('Cumulative infections prediction(England)')
plt.xticks(np.arange(0, flag1 + predict_length + 1, 15), [df.index[i] for i in np.arange(0, flag1 + predict_length + 1, 15)], rotation=30)
plt.xlabel('Day')
plt.ylabel('Cumulative Confirmed Cases')
plt.savefig('./Images/1.png')
plt.close()

loss = []
for i in range(flag1):
    tt = abs(y[i, 2] - prediction_train[i, 0, 0, 2]) / y[i, 2]
    loss.append(tt)
for i in range(predict_length):
    tt = abs(y[flag1 + i, 2] - test_output[i, 2]) / y[flag1 + i, 2]
    loss.append(tt)
plt.plot(np.arange(0, len(loss)), loss)
plt.xlabel("Date")
plt.ylabel("MSE")
plt.xticks(np.arange(0, flag1 + predict_length + 1, 15), [df.index[i] for i in np.arange(0, flag1 + predict_length + 1, 15)], rotation=30)
plt.savefig('./Images/2.png')
plt.close()







