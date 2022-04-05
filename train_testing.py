from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import prettytable as pt


class CalResult:
    def __init__(self, acc, precision, recall, f1):
        self.acc = acc
        self.precision = precision
        self.recall = recall
        self.f1 = f1

    def get_acc(self):
        return self.acc

    def get_precision(self):
        return self.precision

    def get_recall(self):
        return self.recall

    def get_f1(self):
        return self.f1


def cal_score(pres, actual):
    TP = FP = FN = TN = 0
    for i, pre in enumerate(pres):
        if pre > 0:
            if actual[i] == 1:
                TP = TP + 1
            else:
                FP = FP + 1
        else:
            if actual[i] == 1:
                FN = FN + 1
            else:
                TN = TN + 1
    acc = accuracy_score(actual, pres)
    print('准确率：{}'.format(acc))
    print('TP:{} FP:{} FN:{} TN：{}'.format(TP, FP, FN, TN))
    if TP == 0:
        print('TP is 0')
        print()
        return CalResult(acc, 0, 0, 0)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print('precision: {} \nrecall: {} \nf1: {}'.format(precision, recall, f1))
    print()
    return CalResult(acc, precision, recall, f1)


def print_features(features, columns):
    sorted_fea = sorted(zip(map(lambda x: round(x, 4), features), columns), reverse=True)
    table = pt.PrettyTable(field_names=['feature', 'importance'])
    [table.add_row([fea[1], fea[0]]) for fea in sorted_fea]
    print(table)
