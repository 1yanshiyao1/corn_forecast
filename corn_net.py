import hashlib
import os
import tarfile
import zipfile

import matplotlib.pyplot as plt
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

train_data = pd.read_csv('/Users/yanshiyao/PycharmProjects/PythonProject/corn_forecast/agriyield-2025/train.csv')
test_data = pd.read_csv('/Users/yanshiyao/PycharmProjects/PythonProject/corn_forecast/agriyield-2025/test.csv')
# 异常值处理：(因为所有数据均为正常值所以不进行操作)
# train_data['soil_ph']=train_data['soil_ph'].clip(4.0,9.0)


all_features = pd.concat([train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]])

# 构造交互特征
# 土壤特性交互
# soil_ph × organic_matter：土壤 pH 值影响有机质分解效率（酸性 / 碱性过强会抑制微生物活动），两者乘积反映土壤肥力有效性。
# sand_pct × rainfall：砂含量高的土壤排水快，需更多降雨才能满足玉米需求，该交互项反映 “有效水分”（砂含量高→降雨需求高）。
# organic_matter / sand_pct：有机质提升保肥保水能力，砂含量提升排水能力，比值反映 “土壤综合肥力”（值越高，保肥能力越强）。
# 气候因素交互
# temperature × humidity：湿热指数（高温高湿易引发玉米病害，如大斑病），乘积越大，减产风险越高。
# rainfall / temperature：水分 - 热量平衡（温度高则蒸发快，需更多降雨补偿），比值过低可能导致干旱胁迫。
# (temperature - 25)²：玉米最适生长温度约 25°C，偏离越远（过高 / 过低）对产量影响越大，平方项捕捉这种非线性关系。
# 植被与环境交互
# ndvi × temperature：NDVI 反映植被生长状况，其与温度的交互可体现 “温度对植被生长的实际影响”（如高温下 NDVI 高说明抗热品种）。
# ndvi × organic_matter：植被生长受土壤肥力支持，乘积反映 “土壤 - 植被匹配度”（高有机质 + 高 NDVI 通常对应高产）。
# 土壤特性交互：soil_ph × organic_matter；sand_pct × rainfall；organic_matter / sand_pct
all_features['soil_organic'] = all_features['soil_ph'] * all_features['organic_matter']
all_features['sand_rain'] = all_features['sand_pct'] * all_features['rainfall']
all_features['organic_sand'] = all_features['organic_matter'] / all_features['sand_pct']

# 气候因素交互：temperature × humidity；rainfall / temperature；(temperature - 25)²
all_features['tem_hum'] = all_features['temperature'] * all_features['humidity']
all_features['rain_tem'] = all_features['rainfall'] / all_features['temperature']
all_features['tem_2'] = (all_features['temperature'] - 25) ** 2
# 植被与环境交互：ndvi × temperature；ndvi × organic_matter
all_features['ndvi_tem'] = all_features['ndvi'] * all_features['temperature']
all_features['ndvi_organic'] = all_features['ndvi'] * all_features['organic_matter']

# 非线性转换
# 对数转换
all_features['rainfall'] = np.log1p(all_features['rainfall'])  # log(1+x)避免0值问题
# 平方开放转换
all_features['temperature'] = (all_features['temperature']) ** 2
all_features['ndvi'] = (all_features['humidity']) ** 2

# 特征选择：保留高价值特征
all_features = all_features.drop('humidity', axis=1, inplace=False)
print(all_features.shape[1])  # 15个特征

n_train = train_data.shape[0]

# 对所有特征标准化（基于训练集的均值和标准差，避免数据泄露）
numeric_features = all_features.columns
train_features_0 = all_features[:n_train][numeric_features]
# 计算训练集的均值和标准差
mean = train_features_0.mean()
std = train_features_0.std()
# 标准化所有特征（训练集和测试集用同一套参数）
all_features[numeric_features] = (all_features[numeric_features] - mean) / std

train_np = all_features[:n_train].values.astype(np.float32)
test_np = all_features[n_train:].values.astype(np.float32)

train_features = torch.tensor(train_np, dtype=torch.float32)
test_features = torch.tensor(test_np, dtype=torch.float32)
train_labels = torch.tensor(train_data.iloc[:, -1], dtype=torch.float32)

loss = nn.MSELoss()
in_features = train_features.shape[1]  # 列数
print(in_features)


def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),

    )
    return net


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))  # torch.clamp:截断预测值，将模型的预测值小于1则提升为1，大于一则不变
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))  # 均方根误差rmse计算

    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):  # i:当前第几折
    assert k > 1  # 断言语句：只有k大于1才继续向下执行
    fold_size = X.shape[0] // k  # 每一折的大小：样本数/k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # 将每一折里面的取出来
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part  # 把当前折作为验证集
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid  # 返回当前第几折的训练集和验证集


# 返回训练和验证误差的平均值
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]  # 获取最后一个元素加上
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch', ylabel='rmse',
                     xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, 'f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,learning_rate,weight_decay,batch_size):
    net=get_net()
    train_ls,_=train(net,train_features,train_labels,None,None,num_epochs,learning_rate,weight_decay,batch_size)
    d2l.plot(np.arange(1, num_epochs+1), [train_ls],xlabel='epoch',ylabel='log_rems',xlim=[1,num_epochs],yscale='log')
    plt.show()
    print(f'训练log rmse：{float(train_ls[-1]):f}')

    preds=net(test_features).detach().numpy()

    #将其重新格式化以导出到Kaggle

    test_data['yield'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['field_id'], test_data['yield']], axis=1)
    submission.to_csv('sample_submission.csv', index=False)

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.05, 0, 16
# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
# print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 'f'平均验证log rmse: {float(valid_l):f}')
train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size)
