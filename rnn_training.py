import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import dataset, DataLoader, WeightedRandomSampler
from focalloss import FocalLoss
import torch.nn.functional as F
from data_cleaning import *
from train_testing import cal_score
from rnn_model import LSTM

opt_batch_size = 20
opt_hidden_size = 96
opt_num_layers = 2
opt_train_rate = 0.8  # 训练比例
opt_lr = 1e-3  # 学习率
data_path = '/Users/tengyujia/local-data/ai-smart/rnn_train.csv'

opt_batch_first = True


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
    # print(source.head(2))
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
    sorted_ys = torch.FloatTensor([int(ys[x]) for x in perm_idx])
    # 数据按顺序调整
    sorted_xs = [xs[x] for x in perm_idx]
    # padding
    pad_sorted_xs = nn.utils.rnn.pad_sequence(sorted_xs, batch_first=opt_batch_first)
    return pad_sorted_xs, sorted_seq_lengths, sorted_ys


def sigmoid_class(output):
    classes = [0 if d < 0.5 else 1 for d in output]
    return torch.stack((output, torch.LongTensor(classes)), dim=1)


pd_config()
positive_data, negative_data = get_train_dataset(data_path)  # 正例,反例

positive_split_idx = int(positive_data.size * opt_train_rate)
negative_split_idx = int(negative_data.size * opt_train_rate)

train_x = numpy.append(positive_data[:positive_split_idx], negative_data[:negative_split_idx])
train_y = numpy.append(numpy.zeros(positive_split_idx, dtype=float), numpy.ones(negative_split_idx, dtype=float))

train_dataset = TrainData(train_x, train_y)

data_weight = torch.cat((
    torch.ones(positive_split_idx) * (1 / positive_split_idx),
    torch.ones(negative_split_idx) * (1 / negative_split_idx)), 0)

weight_sampler = WeightedRandomSampler(data_weight, len(data_weight), replacement=True)

train_data_loader = DataLoader(train_dataset, batch_size=opt_batch_size, collate_fn=collate_fn,
                               sampler=weight_sampler)

model = LSTM(train_x[0].size()[1], opt_hidden_size, 1, opt_num_layers, opt_batch_first)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr)

# 开始训练
for epoch, (data, lengths, target) in enumerate(train_data_loader):
    # optimizer.zero_grad()
    # 前向传播
    data, target = Variable(data), Variable(target)

    out = model(data, lengths)
    # sigmoid_output = sigmoid_class(out)
    # 计算损失
    loss = criterion(out, target)
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('targets:{}', target)
        print('训练集 epoch:{} Loss: {:.5f}'.format(epoch, loss.data))

    # print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))

model = model.eval()  # 转换成测试模式

test_x = numpy.append(positive_data[positive_split_idx:], negative_data[negative_split_idx:])
test_y = numpy.append(numpy.zeros(positive_data.size - positive_split_idx, dtype=float),
                      numpy.ones(negative_data.size - negative_split_idx, dtype=float))
test_dataset = TrainData(test_x, test_y)
test_data_loader = DataLoader(test_dataset, batch_size=opt_batch_size, collate_fn=collate_fn, shuffle=True)

# 开始 test
# test_data, test_lengths, test_target = collate_fn(test_dataset)
# pred_test = model(test_data, test_lengths)
# pred_test = pred_test.view(-1).data.numpy()
# for t, p in zip(test_target, pred_test):
#     print(t, p)
test_pres = numpy.array([])
test_targets = numpy.array([])
losses = []
for epoch, (data, lengths, target) in enumerate(test_data_loader):
    pred_test = model(data, lengths)
    # sigmoid_output = sigmoid_class(pred_test)
    loss = criterion(pred_test, target)
    # pre_score, pre_targets = torch.split(pred_test, 1, dim=1)
    test_pres = numpy.append(test_pres, pred_test.detach().numpy())
    test_targets = numpy.append(test_targets, target.detach().numpy())
    losses.append(loss.data.item())
    if epoch % 10 == 0:
        print('测试集 epoch:{} Loss: {:.5f}'.format(epoch, loss.data))

print('losses:{}', losses)
cal_score(test_pres.astype(int), test_targets.astype(int))
