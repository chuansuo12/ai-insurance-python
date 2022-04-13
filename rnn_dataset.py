from torch.utils.data import dataset, DataLoader, WeightedRandomSampler
from data_cleaning import *
import torch
import torch.nn as nn
import numpy


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
    id_target = source[['personId', 'classTarget']]
    source.drop(columns=['personId', 'classTarget'], inplace=True)
    source = source.astype('float64')
    low_std_columns = source.std().sort_values().head(35)
    source.drop(columns=low_std_columns.index, inplace=True)
    source = (source - source.mean()) / (source.max() - source.min())  # 归一化
    source = pd.concat([pd.DataFrame(source), id_target], axis=1)
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
    sorted_ys = torch.LongTensor([int(ys[x]) for x in perm_idx])
    # 数据按顺序调整
    sorted_xs = [xs[x] for x in perm_idx]
    # padding
    pad_sorted_xs = nn.utils.rnn.pad_sequence(sorted_xs, batch_first=True)
    return pad_sorted_xs, sorted_seq_lengths, sorted_ys


def get_data_loader(data_path, batch_size, train_rate):
    pd_config()
    positive_data, negative_data = get_train_dataset(data_path)  # 正例,反例

    positive_split_idx = int(positive_data.size * train_rate)
    negative_split_idx = int(negative_data.size * train_rate)
    # 训练数据
    train_x = numpy.append(positive_data[:positive_split_idx], negative_data[:negative_split_idx])
    train_y = numpy.append(numpy.zeros(positive_split_idx, dtype=float), numpy.ones(negative_split_idx, dtype=float))

    train_dataset = TrainData(train_x, train_y)

    data_weight = torch.cat((
        torch.ones(positive_split_idx) * (1 / positive_split_idx),
        torch.ones(negative_split_idx) * (1 / (negative_split_idx * 2))), 0)

    weight_sampler = WeightedRandomSampler(data_weight, len(data_weight), replacement=True)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                   sampler=weight_sampler)
    # 测试数据
    test_x = numpy.append(positive_data[positive_split_idx:], negative_data[negative_split_idx:])
    test_y = numpy.append(numpy.zeros(positive_data.size - positive_split_idx, dtype=float),
                          numpy.ones(negative_data.size - negative_split_idx, dtype=float))
    test_dataset = TrainData(test_x, test_y)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    return train_data_loader, test_data_loader


def get_dataset(data_path, train_rate):
    pd_config()
    positive_data, negative_data = get_train_dataset(data_path)  # 正例,反例

    positive_split_idx = int(positive_data.size * train_rate)
    negative_split_idx = int(negative_data.size * train_rate)
    # 训练数据
    train_x = numpy.append(positive_data[:positive_split_idx], negative_data[:negative_split_idx])
    train_y = numpy.append(numpy.zeros(positive_split_idx, dtype=float), numpy.ones(negative_split_idx, dtype=float))

    train_dataset = TrainData(train_x, train_y)

    data_weight = torch.cat((
        torch.ones(positive_split_idx) * (1 / positive_split_idx),
        torch.ones(negative_split_idx) * (1 / (negative_split_idx * 2))), 0)

    weight_sampler = WeightedRandomSampler(data_weight, len(data_weight), replacement=True)
    # 测试数据
    test_x = numpy.append(positive_data[positive_split_idx:], negative_data[negative_split_idx:])
    test_y = numpy.append(numpy.zeros(positive_data.size - positive_split_idx, dtype=float),
                          numpy.ones(negative_data.size - negative_split_idx, dtype=float))
    test_dataset = TrainData(test_x, test_y)
    return train_dataset, test_dataset, weight_sampler


def get_verify_data(data_path):
    positive_data, negative_data = get_train_dataset(data_path)
    positive_split_idx = int(positive_data.size * 0.8)
    negative_split_idx = int(negative_data.size * 0.8)

    positive_idx = int(positive_data.size * 0.9)
    negative_idx = int(negative_data.size * 0.9)

    verify_x = numpy.append(positive_data[positive_split_idx:positive_idx],
                            negative_data[negative_split_idx:negative_idx])

    verify_y = numpy.append(numpy.zeros(positive_idx - positive_split_idx, dtype=float),
                            numpy.ones(negative_idx - negative_split_idx, dtype=float))
    return TrainData(verify_x, verify_y)


def get_combine_test_data(data_path):
    positive_data, negative_data = get_train_dataset(data_path)

    positive_idx = int(positive_data.size * 0.9)
    negative_idx = int(negative_data.size * 0.9)

    test_x = numpy.append(positive_data[positive_idx:],
                          negative_data[negative_idx:])

    test_y = numpy.append(numpy.zeros(positive_data.size - positive_idx, dtype=float),
                          numpy.ones(negative_data.size - negative_idx, dtype=float))
    return TrainData(test_x, test_y)
