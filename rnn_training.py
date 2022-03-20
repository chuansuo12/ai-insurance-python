import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import dataset, DataLoader

from data_cleaning import *

batch_size = 2
opt_batch_first = True


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(LSTM, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, batch_first=opt_batch_first)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, length_list):
        x = nn.utils.rnn.pack_padded_sequence(x, lengths=length_list, batch_first=opt_batch_first)
        x, _ = self.layer1(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


class TrainData(dataset.Dataset):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

    def __len__(self):
        return len(self.dataset_x)

    def __getitem__(self, item):
        return self.dataset_x[item], self.dataset_y[item]


def trans_to_tensor(dt=DataFrame):
    nps = dt.drop(labels='personId', axis=1, inplace=False).to_numpy()
    return torch.FloatTensor(nps)


def get_train_test(file_path):
    source = read_file(file_path)
    print(source.head(2))
    y = source['classTarget'].groupby(by=source['personId']).agg('max').to_numpy()
    y = y.astype(int)
    source['personId'] = source['personId'] - 352120000000000  # 防止ID过长
    source = source.astype('float32')
    source = (source - source.min()) / (source.max() - source.min())  # 归一化
    source.fillna(value=0, inplace=True)
    x = source.drop(labels='classTarget', axis=1, inplace=False) \
        .groupby(source['personId']).apply(trans_to_tensor).values
    return x, y


def collate_fn(batch):
    xs = [v[0] for v in batch]
    ys = [v[1] for v in batch]
    # 获得每个样本的序列长度
    seq_lengths = torch.LongTensor([v for v in map(len, xs)])
    # 样本长度排序
    sorted_seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    # 标签按顺序调整
    sorted_ys = torch.FloatTensor([ys[idx] for idx in perm_idx])
    # 数据按顺序调整
    sorted_xs = [xs[idx] for idx in perm_idx]
    # padding
    pad_sorted_xs = nn.utils.rnn.pad_sequence(sorted_xs, batch_first=opt_batch_first)
    return pad_sorted_xs, sorted_seq_lengths, sorted_ys


data_path = '/Users/tengyujia/local-data/ai-smart/'

pd_config()
train_x, train_y = get_train_test(data_path + '/rnn_test.csv')
train_dataset = TrainData(train_x, train_y)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)

model = LSTM(train_x[0].size()[1], 4, 1, 2)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 开始训练
for data, lengths, target in train_data_loader:
    # 前向传播
    out = model(data, lengths)

    loss = criterion(out, target)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Loss: {:.5f}'.format(loss.data))

    # print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))

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
