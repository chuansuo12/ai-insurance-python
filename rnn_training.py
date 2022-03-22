import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import dataset, DataLoader
import torch.nn.functional as F
from data_cleaning import *

batch_size = 10
train_rate = 0.8  # 训练比例
data_path = '/Users/tengyujia/local-data/ai-smart/rnn_test.csv'

opt_batch_first = True


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first=opt_batch_first)
        self.fc = nn.Linear(hidden_size, output_size)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net(self, x):  # x:[batch, seq_len, hidden_dim]
        # 原文中使用了 dropout 此处弃用 https://www.cnblogs.com/cxq1126/p/13504437.html
        u = torch.tanh(torch.matmul(x, self.w_omega))  # [batch, seq_len, hidden_dim]
        att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score  # [batch, seq_len, hidden_dim]
        context = torch.sum(scored_x, dim=1)  # [batch, hidden_dim]
        return context

    def forward(self, x, length_list):
        # 解决同一 batch 数据变长问题
        x = nn.utils.rnn.pack_padded_sequence(x, lengths=length_list, batch_first=opt_batch_first)
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        x = x.permute(1, 0, 2)  # [batch, seq_len, hidden_dim]
        attn_output = self.attention_net(x)
        s = torch.sigmoid(attn_output)
        # logit = self.fc(attn_output)
        # torch.sigmoid(torch.sum(attn_output, dim=1))
        return torch.sigmoid(torch.sum(attn_output, dim=1))


class TrainData(dataset.Dataset):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

    def __len__(self):
        return len(self.dataset_x)

    def __getitem__(self, item):
        return self.dataset_x[item], self.dataset_y[item]


def trans_to_tensor(dt=DataFrame):
    nps = dt.drop(labels='personId', axis=1, inplace=False)
    nps.sort_values(by='transactionHour', inplace=True, ignore_index=True)
    return torch.FloatTensor(nps.to_numpy())


def group_by_person(dt=DataFrame):
    tg = dt['classTarget'].max()
    dt = dt.drop(labels='classTarget', axis=1, inplace=False)
    return [tg, dt.groupby(dt['personId']).apply(trans_to_tensor).values]  # [target,list(data)]


def get_train_dataset(file_path):
    source = read_file(file_path)
    print(source.head(2))
    source['personId'] = source['personId'] - 352120000000000  # 防止ID过长
    source = source.astype('float32')
    source = (source - source.min()) / (source.max() - source.min())  # 归一化
    source.fillna(value=0, inplace=True)
    x = source.groupby(source['classTarget']).apply(group_by_person).values
    x = sorted(x, key=lambda item: item[0])
    return x[0][1], x[1][1]  # target=0,target=1


def collate_fn(batch):
    xs = [v[0] for v in batch]
    ys = [v[1] for v in batch]
    # 获得每个样本的序列长度
    seq_lengths = torch.LongTensor([v for v in map(len, xs)])
    # 样本长度排序
    sorted_seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    # 标签按顺序调整
    sorted_ys = torch.FloatTensor([ys[x] for x in perm_idx])
    # 数据按顺序调整
    sorted_xs = [xs[x] for x in perm_idx]
    # padding
    pad_sorted_xs = nn.utils.rnn.pad_sequence(sorted_xs, batch_first=opt_batch_first)
    return pad_sorted_xs, sorted_seq_lengths, sorted_ys


def sigmoid_class(output):
    classes = [0 if d < 0.5 else 1 for d in output]
    return torch.stack((output, torch.LongTensor(classes)), dim=1)


pd_config()
positive_data, negative_data = get_train_dataset(data_path)

positive_split_idx = int(positive_data.size * train_rate)
negative_split_idx = int(negative_data.size * train_rate)

train_x = numpy.append(positive_data[:positive_split_idx], negative_data[:negative_split_idx])
train_y = numpy.append(numpy.zeros(positive_split_idx, dtype=float), numpy.ones(negative_split_idx, dtype=float))

train_dataset = TrainData(train_x, train_y)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

model = LSTM(train_x[0].size()[1], 4, 1, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 开始训练
for idx, (data, lengths, target) in enumerate(train_data_loader):
    # 前向传播
    out = model(data, lengths)
    sigmoid_output = sigmoid_class(out)
    loss = criterion(sigmoid_output, target)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Loss: {:.5f}'.format(loss.data))

    # print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))

model = model.eval()  # 转换成测试模式

test_x = numpy.append(positive_data[positive_split_idx:], negative_data[negative_split_idx:])
test_y = numpy.append(numpy.zeros(positive_data.size - positive_split_idx, dtype=float),
                      numpy.ones(negative_data.size - negative_split_idx, dtype=float))
test_dataset = TrainData(test_x, test_y)
# test_data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

# 开始 test
test_data, test_lengths, test_target = collate_fn(test_dataset)
pred_test = model(test_data, test_lengths)
pred_test = pred_test.view(-1).data.numpy()
for t, p in zip(test_target, pred_test):
    print(t, p)

# for data, lengths, target in test_data_loader:
#     pred_test = model(data, lengths)
#     pred_test = pred_test.view(-1).data.numpy()
#     print(target, pred_test)
