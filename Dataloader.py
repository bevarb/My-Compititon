import pandas as pd
import numpy as np
import torch
# 数据预处理
def get_train_test(root, begain, over, nums, N):
    df = pd.read_csv(root)
    df.set_index(["Date"], inplace=True)


    # Exposed = df.loc["1/22/20":"5/15/20", "Exposed"]
    Removed = df.loc[begain:over, "Recovered"] + df.loc[begain:over, "Dead"]
    Confirmed = df.loc[begain:over, "Confirmed"] - Removed
    Exposed = Confirmed * 0.7
    Suceptible = np.ones(len(Confirmed)) * nums - Confirmed - Exposed - Removed

    All_value = pd.concat([Suceptible, Exposed], axis=1)
    All_value = pd.concat([All_value, Confirmed], axis=1)
    All_value = pd.concat([All_value, Removed], axis=1)
    All_value = np.array(All_value)

    TRAIN_SIZE = 0.7

    x = []
    y = []
    seq = 1  # 选择几天的数据作为小输入
    feat_nums = 1
    for i in range(len(All_value)-seq-1):
        x.append(All_value[i:i+seq][:])
        y.append(All_value[i+seq][:])

    Data_Length = len(y)
    flag1 = int(TRAIN_SIZE * Data_Length)   # 训练集与测试集的分界线


    train_x = (torch.tensor(x[0:flag1]).float()/N)
    train_y = (torch.tensor(y[0:flag1]).float()/N)
    test_x = (torch.tensor(x[flag1:Data_Length]).float()/N)
    test_y = (torch.tensor(y[flag1:Data_Length]).float()/N)

    return x, y, train_x, train_y, test_x, test_y, Data_Length, flag1

