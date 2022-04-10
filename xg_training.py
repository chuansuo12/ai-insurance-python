import xgboost as xgb
from pandas.core.frame import DataFrame
from train_testing import *


def xgb_config():
    xgb.set_config(verbosity=2)


def xgb_train(train_data=DataFrame, test_data=DataFrame):
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',  # 二分类概率
        'num_class': 2,  # 类别数，与 multisoftmax 并用
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 8,  # 构建树的深度，越大越容易过拟合
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,  # 随机采样训练样本
        'colsample_bytree': 0.7,  # 生成树时进行的列采样
        'min_child_weight': 3,
        'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.007,  # 如同学习率
        'seed': 1000,
        'nthread': 10,  # cpu 线程数
    }
    train_matrix = get_matrix(train_data)
    test_matrix = get_matrix(test_data)
    evallist = [(test_matrix, 'eval'), (train_matrix, 'train')]
    num_round = 10
    bst = xgb.train(params, train_matrix, num_round, evallist)
    return bst


def get_matrix(data=DataFrame):
    label = data[['classTarget']]
    return xgb.DMatrix(data.drop(columns=['classTarget']), label=label)


def read_model(model_path):
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def xgb_test(test_data, booster):
    label = test_data['classTarget']
    test_matrix = get_matrix(test_data)
    preds = booster.predict(test_matrix)
    cal_score(preds, label.values)
