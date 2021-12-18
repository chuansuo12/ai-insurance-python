from sklearn.ensemble import GradientBoostingClassifier
from train_testing import *
from pandas import DataFrame


def gbdt_train(train_data=DataFrame):
    x = get_x(train_data)
    y = get_y(train_data)
    gbdt = GradientBoostingClassifier(random_state=10, max_depth=3)
    gbdt.fit(x, y)
    return gbdt


def gbdt_test(gbdt=GradientBoostingClassifier, test_data=DataFrame):
    x = get_x(test_data)
    y_pre = gbdt.predict(x)
    y = get_y(test_data)
    cal_score(y_pre, y)


def get_x(data=DataFrame):
    return data.drop(columns=['classTarget'])


def get_y(data=DataFrame):
    return data['classTarget'].tolist()
