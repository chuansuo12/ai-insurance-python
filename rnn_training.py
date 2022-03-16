import torch
import torch.nn as nn
from pandas import DataFrame
from data_cleaning import *
from datetime import *
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(LSTM, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


def trans_to_numpy(data=DataFrame):
    return data.drop(labels='personId', axis=1, inplace=False).to_numpy()


def get_train_test(file_path):
    source = read_file(file_path)
    print(source.head(10))
    y = source['classTarget'].groupby(by=source['personId']).agg('max').to_numpy()
    x = source.drop(labels='classTarget', axis=1, inplace=False) \
        .groupby(source['personId']).apply(trans_to_numpy)
    return x, y


data_path = '/Users/tengyujia/local-data/ai-smart/'

pd_config()
train_x, train_y = get_train_test(data_path + '/rnn_test.csv')

model = LSTM(2, 4, 1, 2)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 开始训练
for e in range(1000):
    var_x = Variable(torch.from_numpy(train_x))
    var_y = Variable(torch.from_numpy(train_y))
    # 前向传播
    out = model(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e + 1) % 100 == 0:  # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))

model = model.eval()  # 转换成测试模式
test_x, test_y = get_train_test(data_path + '/rnn_train.csv')

var_data = Variable(test_x)
pred_test = model(var_data)  # 测试集的预测结果
# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()
print(test_y, pred_test)
#
# rnn = nn.LSTM(10, 20, 2)  # 一个单词向量长度为10，隐藏层节点数为20，LSTM有2层
# input = torch.randn(5, 3, 10)  # 输入数据由3个句子组成，每个句子由5个单词组成，单词向量长度为10
# print(input)
# h0 = torch.randn(2, 3, 20)  # 2：LSTM层数*方向 3：batch 20： 隐藏层节点数
# c0 = torch.randn(2, 3, 20)  # 同上
# output, (hn, cn) = rnn(input, (h0, c0))
#
# print(output.shape, hn.shape, cn.shape)
#
# print(hn)
