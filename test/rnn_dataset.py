import pandas as pd
from torch.utils.data import dataset, DataLoader, WeightedRandomSampler
from data_cleaning import *
import torch
import torch.nn as nn
import numpy


def trans_to_tensor(dt=DataFrame):
    return dt['personId'].unique()[0]


def group_by_person(dt=DataFrame):
    tg = dt['classTarget'].max()
    dt = dt.drop(labels='classTarget', axis=1, inplace=False)
    return [tg, dt.groupby(dt['personId']).apply(trans_to_tensor).values]  # [target,list(data)]


def get_train_dataset(file_path):
    source = read_file(file_path)
    # print(source.head(2))
    # source['personId'] = source['personId'] - 352120000000000  # 防止ID过长
    # id_target = source[['personId', 'classTarget']]
    # source.drop(columns=['personId', 'classTarget'], inplace=True)
    # source = source.astype('float64')
    # low_std_columns = source.std().sort_values().head(35)
    # source.drop(columns=low_std_columns.index, inplace=True)
    # source = (source - source.mean()) / (source.max() - source.min())  # 归一化
    # source = pd.concat([pd.DataFrame(source), id_target], axis=1)
    # source.fillna(value=0, inplace=True)
    # source = pd.DataFrame(source, columns=['classTarget', 'personId', 'transactionHour'])
    x = source.groupby(source['classTarget']).apply(group_by_person).values
    x = sorted(x, key=lambda item: item[0])
    return x[0][1], x[1][1]  # target=0,target=1

def split_ids():
    id_dir = '/Users/tengyujia/local-data/ai-smart/train_ids/'
    positive_data, negative_data = get_train_dataset('/Users/tengyujia/local-data/ai-smart/rnn_train_v2.csv')
    positive_split_idx = int(positive_data.size * 0.8)
    negative_split_idx = int(negative_data.size * 0.8)

    train_positive_ids = positive_data[:positive_split_idx]
    train_negative_positive_ids = negative_data[:negative_split_idx]

    train_ids = pd.DataFrame(numpy.append(train_positive_ids, train_negative_positive_ids))
    train_ids.to_csv(id_dir + '/train_ids.csv', index=False)

    positive_idx = int(positive_data.size * 0.9)
    negative_idx = int(negative_data.size * 0.9)

    verify_positive_ids = positive_data[positive_split_idx:positive_idx]
    verify_negative_positive_ids = negative_data[negative_split_idx:negative_idx]

    verify_ids = pd.DataFrame(numpy.append(verify_positive_ids, verify_negative_positive_ids))
    verify_ids.to_csv(id_dir + '/verify_ids.csv', index=False)

    test_positive_ids = positive_data[positive_idx:]
    test_negative_positive_ids = negative_data[negative_idx:]

    test_ids = pd.DataFrame(numpy.append(test_positive_ids, test_negative_positive_ids))
    test_ids.to_csv(id_dir + '/test_ids.csv', index=False)

def split_test():
    id_dir = '/Users/tengyujia/local-data/ai-smart'
    positive_data, negative_data = get_train_dataset('/Users/tengyujia/local-data/ai-smart/rnn_train_v2.csv')


if __name__ == '__main__':
    split_ids()
