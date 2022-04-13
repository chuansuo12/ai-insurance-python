import matplotlib.pyplot as plt
import numpy
import pandas as pd
import torch
import torch.nn as nn
from rnn_dataset import get_data_loader, get_dataset, collate_fn, DataLoader
from rnn_model import LSTM
from train_testing import cal_score
from rnn_model_v2 import Encoder, Attention, Classifier, ModelConfig
import time
from data_cleaning import *

opt_batch_size = 2
opt_hidden_size = 64
opt_num_layers = 2
opt_train_rate = 0.8  # 训练比例
opt_lr = 1e-3  # 学习率
opt_drop_out = 0
opt_bidirectional = False  # 双向 LSTM
opt_model = 'LSTM'  # LSTM OR GRU
opt_gradient_clipping = 5
opt_seed = 3

dir_path = '/Users/tengyujia/local-data/ai-smart/'
model_path = dir_path + 'rnn_models/'
data_path = dir_path + 'rnn_test_part.csv'

opt_batch_first = True
opt_output = True


def sigmoid_class(output):
    classes = [0 if d < 0.5 else 1 for d in output]
    return torch.stack((output, torch.LongTensor(classes)), dim=1)


def update_stats(accuracy, confusion_matrix, predict, y):
    _, max_ind = torch.max(predict, 1)
    equal = torch.eq(max_ind, y)
    correct = int(torch.sum(equal))

    for i, j in zip(y, max_ind):
        confusion_matrix[int(i), int(j)] += 1

    return accuracy + correct, confusion_matrix


def train(model, data, optimizer, criterion, clip):
    model.train()
    accuracy, confusion_matrix = 0, numpy.zeros((2, 2), dtype=int)
    t = time.time()
    total_loss = 0
    epochs = []
    losses = []
    for epoch, (part_data, lengths, target) in enumerate(data):
        model.zero_grad()

        lgs, _ = model(part_data, lengths)
        loss = criterion(lgs.view(-1, 2), target)
        epochs.append(epoch)
        losses.append(loss.data)
        total_loss += float(loss)
        accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, lgs, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # print("[Batch]: {}/{} in {:.5f} seconds".format(
        #    epoch, len(data.dataset) / opt_batch_size, time.time() - t))
        t = time.time()
    print("[Loss]: {:.5f}".format(total_loss / len(data)))
    print("[Accuracy]: {}/{} : {:.3f}%".format(
        accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
    print(confusion_matrix)
    # plt.plot(epochs, losses)
    # plt.show()
    return total_loss / len(data)


# 开始 test
# test_data, test_lengths, test_target = collate_fn(test_dataset)
# pred_test = model(test_data, test_lengths)
# pred_test = pred_test.view(-1).data.numpy()
# for t, p in zip(test_target, pred_test):
#     print(t, p)
def test(model, criterion, tests):
    model.eval()  # 转换成测试模式
    test_pres = numpy.array([])
    test_targets = numpy.array([])
    accuracy, confusion_matrix = 0, numpy.zeros((2, 2), dtype=int)
    t = time.time()
    total_loss = 0
    with torch.no_grad():
        for epoch, (part_data, lengths, target) in enumerate(tests):
            lgs, _ = model(part_data, lengths)
            total_loss += float(criterion(lgs.view(-1, 2), target))
            _, pred_test = torch.max(lgs, 1)
            # pre_score, pre_targets = torch.split(pred_test, 1, dim=1)
            accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, lgs, target)
            test_pres = numpy.append(test_pres, pred_test.detach().numpy())
            test_targets = numpy.append(test_targets, target.detach().numpy())
            print("[Batch]: {}/{} in {:.5f} seconds".format(
                len(part_data), len(tests.dataset), time.time() - t), end='\r', flush=True)
            t = time.time()

    print()
    print("[Valid loss]: {:.5f}".format(total_loss / len(tests.dataset)))
    # print("[Valid accuracy]: {}/{} : {:.3f}%".format(
    #     accuracy, len(test_data.dataset),
    #     accuracy / len(test_data.dataset) * 100))
    # print(confusion_matrix)
    # if confusion_matrix[1][1] == 0:
    #     return 0
    # recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
    # precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
    # f1_score = 2 * precision * recall / (precision + recall)
    # print("precision:", precision)
    # print("recall:", recall)
    # print("F1:", f1_score)
    print('model:{},bi:{},batch_size:{},hidden_size:{}'.format(
        opt_model, opt_bidirectional, opt_batch_size, opt_hidden_size))
    return cal_score(test_pres.astype(int), test_targets.astype(int))


def seed_everything(seed=opt_seed, cuda=False):
    # Set the random seed manually for reproducibility.
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def epoch_train_test(model_config, train_dataset, test_dataset, sampler):
    train_data_loader = DataLoader(
        train_dataset, batch_size=opt_batch_size, collate_fn=collate_fn, sampler=sampler)
    seed_everything(opt_seed)
    model, criterion, optimizer = get_model(train_data_loader.dataset[0][0].size()[1])
    train(model, train_data_loader, optimizer, criterion, opt_gradient_clipping)
    test_data_loader = DataLoader(
        test_dataset, batch_size=opt_batch_size, collate_fn=collate_fn, shuffle=True)
    result = test(model, criterion, test_data_loader)
    if opt_output:
        torch.save(model, model_path + model_config.get_name() + '_F1=' + str(round(result.get_f1(), 4))
                   + '_b=' + str(opt_batch_size) + '_h=' + str(opt_hidden_size) + '_s=' + str(opt_seed))
    return result


def get_model(input_size):
    encoder = Encoder(
        input_size,
        hidden_size=opt_hidden_size,
        num_layers=opt_num_layers,
        dropout=opt_drop_out,
        bidirectional=opt_bidirectional,
        rnn_type=opt_model)
    attention_dim = opt_hidden_size if not opt_bidirectional else 2 * opt_hidden_size
    attention = Attention(attention_dim, attention_dim, attention_dim)
    model = Classifier(encoder, attention, attention_dim, 2)
    model.to('cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr, amsgrad=True)
    return model, criterion, optimizer


if __name__ == '__main__':
    models = [
        # ModelConfig('LSTM', '^', False, 'LSTM'),
        # ModelConfig('GRU', 's', False, 'GRU'),
        # ModelConfig('Bi-LSTM', 'o', True, 'LSTM'),
        ModelConfig('Bi-GRU', 'o', True, 'GRU')
    ]
    train_data, test_data, weight_sampler = get_dataset(data_path, opt_train_rate)
    # batch_sizes = [2, 4, 8, 16, 32, 64, 128]
    hidden_sizes = [32, 64, 96]
    # max_seed = -1
    # max_f1 = -1
    # for seed in range(0, 100):
    #     opt_seed = seed
    #     f1 = epoch_train_test(ModelConfig('Bi-LSTM', 'o'), train_data, test_data, weight_sampler)
    #     print('seed:{},f1:{}'.format(seed, f1))
    #     if f1 > max_f1:
    #         max_seed = seed
    #         max_f1 = f1
    # print('max_seed:{},max_f1:{}'.format(max_seed, max_f1))
    x_label = 'batch_size'
    plt.figure(layout='constrained')
    plt.xlabel(x_label)
    plt.ylabel('F1')
    # plt.title("hidden size={}".format(opt_hidden_size))
    plt.title("batch size={}".format(opt_batch_size))
    for model_conf in models:
        if model_conf.get_bi():
            opt_bidirectional = True
        else:
            opt_bidirectional = False
        opt_model = model_conf.get_md()
        f1_lst = []
        # if model_conf.get_bi():
        #     opt_hidden_size = 32
        for hidden_size in hidden_sizes:
            # opt_batch_size = batch_size
            if model_conf.get_bi():
                opt_hidden_size = int(hidden_size / 2)
            else:
                opt_hidden_size = hidden_size
            cal_result = epoch_train_test(model_conf, train_data, test_data, weight_sampler)
            f1_lst.append(cal_result.get_f1())
            if opt_output:
                with open(dir_path + 'result_data/model_result.csv', 'a') as result_file:
                    result_file.write('{},{},{},{},{},{},{}\n'.format(
                        model_conf.get_name(),
                        opt_batch_size,
                        opt_hidden_size,
                        round(cal_result.get_acc(), 4),
                        round(cal_result.get_precision(), 4),
                        round(cal_result.get_recall(), 4),
                        round(cal_result.get_f1(), 4),
                    ))
        plt.plot(hidden_sizes, f1_lst, label=model_conf.get_name(),
                 marker=model_conf.get_marker())
    plt.legend()
    plt.show()
