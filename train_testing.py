from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import prettytable as pt


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
    print('准确率：{}'.format(accuracy_score(actual, pres)))
    print('TP:{} FP:{} FN:{} TN：{}'.format(TP, FP, FN, TN))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print('precision: {} \nrecall: {} \nf1: {}'.format(precision, recall, f1))


def print_features(features, columns):
    sorted_fea = sorted(zip(map(lambda x: round(x, 4), features), columns), reverse=True)
    table = pt.PrettyTable(field_names=['feature', 'importance'])
    [table.add_row([fea[1], fea[0]]) for fea in sorted_fea]
    print(table)



