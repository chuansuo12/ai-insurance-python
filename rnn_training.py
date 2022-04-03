import matplotlib.pyplot as plt
import numpy
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from rnn_dataset import get_data_loader
from rnn_model import LSTM
from train_testing import cal_score
from rnn_model_v2 import Encoder, Attention, Classifier, ModelConfig
import time
from data_cleaning import *

opt_batch_size = 20
opt_hidden_size = 64
opt_num_layers = 2
opt_train_rate = 0.8  # 训练比例
opt_lr = 1e-3  # 学习率
opt_drop_out = 0
opt_bidirectional = False  # 双向 LSTM
opt_model = 'LSTM'  # LSTM OR GRU
opt_gradient_clipping = 5

dir_path = '/Users/tengyujia/local-data/ai-smart/'
model_path = dir_path + 'rnn_models/'
data_path = dir_path + 'rnn_test.csv'

opt_batch_first = True


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

        logits, _ = model(part_data, lengths)
        loss = criterion(logits.view(-1, 2), target)
        epochs.append(epoch)
        losses.append(loss.data)
        total_loss += float(loss)
        accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # print("[Batch]: {}/{} in {:.5f} seconds".format(
        #    epoch, len(data.dataset) / opt_batch_size, time.time() - t))
        t = time.time()

    print()
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
def test(model, criterion, test_data):
    model.eval()  # 转换成测试模式
    test_pres = numpy.array([])
    test_targets = numpy.array([])
    model.eval()
    accuracy, confusion_matrix = 0, numpy.zeros((2, 2), dtype=int)
    t = time.time()
    total_loss = 0
    with torch.no_grad():
        for epoch, (part_data, lengths, target) in enumerate(test_data):
            logits, _ = model(part_data, lengths)
            total_loss += float(criterion(logits.view(-1, 2), target))
            _, pred_test = torch.max(logits, 1)
            # pre_score, pre_targets = torch.split(pred_test, 1, dim=1)
            accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, target)
            test_pres = numpy.append(test_pres, pred_test.detach().numpy())
            test_targets = numpy.append(test_targets, target.detach().numpy())
            print("[Batch]: {}/{} in {:.5f} seconds".format(
                len(part_data), len(test_data.dataset), time.time() - t), end='\r', flush=True)
            t = time.time()

    print()
    print("[Valid loss]: {:.5f}".format(total_loss / len(test_data.dataset)))
    print("[Valid accuracy]: {}/{} : {:.3f}%".format(
        accuracy, len(test_data.dataset),
        accuracy / len(test_data.dataset) * 100))
    print(confusion_matrix)
    if confusion_matrix[1][1] == 0:
        return 0
    recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
    precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
    f1_score = 2 * precision * recall / (precision + recall)
    print("precision:", precision)
    print("recall:", recall)
    print("F1:", f1_score)
    # cal_score(test_pres.astype(int), test_targets.astype(int))
    return f1_score


def seed_everything(seed, cuda=False):
    # Set the random seed manually for reproducibility.
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def epoch_train_test(model_config=ModelConfig):
    seed_everything(6)

    encoder = Encoder(
        train_data_loader.dataset[0][0].size()[1],
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
    train(model, train_data_loader, optimizer, criterion, opt_gradient_clipping)
    f1_score = test(model, criterion, test_data_loader)
    torch.save(model, model_path + model_config.get_name() + '_F1=' + str(round(f1_score,4))
               + '_b=' + str(opt_batch_size) + '_h=' + str(opt_hidden_size))
    return f1_score


if __name__ == '__main__':
    models = [ModelConfig('LSTM', '^'),
              ModelConfig('GRU', 's'),
              ModelConfig('Bi-LSTM', 'o')]
    batch_sizes = [5, 10, 15, 20, 25, 30]
    x_label = 'batch_size'
    plt.figure(layout='constrained')
    plt.xlabel(x_label)
    plt.ylabel('F1')
    plt.title("hidden size={}".format(opt_hidden_size))
    train_data_loader, test_data_loader = get_data_loader(data_path, opt_batch_size, opt_train_rate)
    f1_frames = pd.DataFrame()
    f1_frames[x_label] = batch_sizes
    for model_conf in models:
        if 'Bi-LSTM' == model_conf.get_name():
            opt_bidirectional = True
            opt_model = 'LSTM'
        else:
            opt_bidirectional = False
            opt_model = model_conf.get_name()
        f1_lst = []
        for batch_size in batch_sizes:
            opt_batch_size = batch_size
            f1 = epoch_train_test(model_conf)
            f1_lst.append(f1)
        plt.plot(batch_sizes, f1_lst, label=model_conf.get_name(),
                 marker=model_conf.get_marker())
        f1_frames[model_conf.get_name()] = f1_lst

    plt.legend()
    plt.show()
    f1_frames.to_csv(dir_path + 'result_data/' + x_label + '_curve.csv', index=False)
