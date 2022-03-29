import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from rnn_dataset import get_data_loader
from rnn_model import LSTM
from train_testing import cal_score
from rnn_model_v2 import Encoder, Attention, Classifier

opt_batch_size = 10
opt_hidden_size = 64
opt_num_layers = 2
opt_train_rate = 0.8  # 训练比例
opt_lr = 1e-3  # 学习率
opt_drop_out = 0
opt_bidirectional = False  # 双向 LSTM
opt_model = 'LSTM'  # LSTM OR GRU
opt_gradient_clipping = 5

data_path = '/Users/tengyujia/local-data/ai-smart/rnn_train.csv'

opt_batch_first = True


def sigmoid_class(output):
    classes = [0 if d < 0.5 else 1 for d in output]
    return torch.stack((output, torch.LongTensor(classes)), dim=1)


def update_stats(accuracy, confusion_matrix, logits, y):
    _, max_ind = torch.max(logits, 1)
    equal = torch.eq(max_ind, y)
    correct = int(torch.sum(equal))

    for j, i in zip(max_ind, y):
        confusion_matrix[int(i), int(j)] += 1

    return accuracy + correct, confusion_matrix


def train(model, criterion, optimizer, train_data):
    model.train()
    epochs = []
    losses = []
    accuracy, confusion_matrix = 0, numpy.zeros((2, 2), dtype=int)
    # 开始训练
    for epoch, (data, lengths, target) in enumerate(train_data):
        optimizer.zero_grad()
        # 前向传播
        # data, target = Variable(data), Variable(target)
        out, _ = model(data, lengths)
        # sigmoid_output = sigmoid_class(out)
        # 计算损失
        loss = criterion(out.view(-1, 2), target)
        epochs.append(epoch)
        losses.append(loss.data)
        accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, out, target)
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt_gradient_clipping)
        optimizer.step()
        if epoch % 10 == 0:
            print('targets:{}', target)
            print('训练集 epoch:{} Loss: {:.5f}'.format(epoch, loss.data))

    print('Total Loss: {:.5f}'.format(sum(losses)))
    print("[Accuracy]: {}/{} : {:.3f}%".format(
        accuracy, len(train_data.dataset), accuracy / len(train_data.dataset) * 100))
    plt.plot(epochs, losses)
    print(confusion_matrix)
    plt.show()


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
    losses = []
    accuracy, confusion_matrix = 0, numpy.zeros((2, 2), dtype=int)
    with torch.no_grad():
        for epoch, (data, lengths, target) in enumerate(test_data):
            out, _ = model(data, lengths)
            # sigmoid_output = sigmoid_class(pred_test)
            loss = criterion(out, target)
            _, pred_test = torch.max(out, 1)
            # pre_score, pre_targets = torch.split(pred_test, 1, dim=1)
            accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, out, target)
            test_pres = numpy.append(test_pres, pred_test.detach().numpy())
            test_targets = numpy.append(test_targets, target.detach().numpy())
            losses.append(loss.data.item())
            if epoch % 10 == 0:
                print('测试集 epoch:{} Loss: {:.5f}'.format(epoch, loss.data))
        print()
        print("[Valid loss]: {:.5f}".format(sum(losses)))
        print("[Valid accuracy]: {}/{} : {:.3f}%".format(
            accuracy, len(test_data.dataset),
            accuracy / len(test_data.dataset) * 100))
        print(confusion_matrix)
        precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
        recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
        print("precision:", precision)
        print("recall:", recall)
        print("F1:", 2 * precision * recall / (precision + recall))

    print('losses:', losses)
    cal_score(test_pres.astype(int), test_targets.astype(int))


if __name__ == '__main__':
    train_data_loader, test_data_loader = get_data_loader(data_path, opt_batch_size, opt_train_rate)
    # model = LSTM(train_data_loader.dataset[0][0].size()[1], opt_hidden_size, 1, opt_num_layers, opt_batch_first,
    #              opt_batch_size)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr, amsgrad=True)
    train(model, criterion, optimizer, train_data_loader)
    test(model, criterion, test_data_loader)
