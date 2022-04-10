import sklearn.metrics
import torch
from torch.utils.data import DataLoader

from gbdt_training import *
from data_cleaning import *
from sklearn.utils import shuffle
import numpy as np
import joblib
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from rnn_dataset import get_dataset, collate_fn, get_verify_data
from rnn_training import get_model, seed_everything

data_path = '/Users/tengyujia/local-data/ai-smart/'

gbdt_path = data_path + 'combine/gbdt.model'
lstm_model = data_path + 'combine/LSTM_F1=0.4622_b=2_h=64_s=3'
pd_config()
opt_train_rate = 0.8


def train_gbdt():
    np.random.seed(3)
    train_ids = read_file(data_path + 'combine/train_ids.csv')
    verify_ids = read_file(data_path + 'combine/verify_ids.csv')
    person_data = read_file(data_path + 'person_group.csv')
    train_data = pd.DataFrame(person_data[person_data['personId'].isin(train_ids.values.squeeze())])
    verify_data = pd.DataFrame(person_data[person_data['personId'].isin(verify_ids.values.squeeze())])
    train_data.drop(columns='personId', inplace=True)
    verify_data.drop(columns='personId', inplace=True)
    train_data = shuffle(train_data)
    gbdt_local = gbdt_train(train_data)
    gbdt_test(gbdt_local, verify_data)
    feature_importance = list(gbdt_local.feature_importances_)
    print_features(feature_importance, train_data.columns)
    joblib.dump(gbdt_local, gbdt_path)


if __name__ == '__main__':
    # train_gbdt()
    seed_everything()
    verify_ids = read_file(data_path + 'combine/verify_ids.csv')
    person_data = read_file(data_path + 'person_group.csv')
    verify_data = pd.DataFrame(person_data[person_data['personId'].isin(verify_ids.values.squeeze())])
    verify_dataset = get_verify_data(data_path + 'rnn_train_v2.csv')
    test_data_loader = DataLoader(verify_dataset, batch_size=1, collate_fn=collate_fn)
    model = torch.load(lstm_model)
    gbdt_local = joblib.load(gbdt_path)
    for i in range(1, 100, 5):
        era = i * 0.01
        test_pres = np.array([])
        test_targets = np.array([])
        with torch.no_grad():
            for idx, verify_id in enumerate(verify_ids.values):
                verify_id = verify_id[0]
                cur_data = pd.DataFrame(verify_data[verify_data['personId'] == verify_id])
                cur_data.drop(columns='personId', inplace=True)
                rate0 = gbdt_test_per(gbdt_local, cur_data)
                lgs, _ = model(test_data_loader.dataset[idx][0].unsqueeze(0),
                               [test_data_loader.dataset[idx][0].shape[0]])
                target = era * rate0[0] + (1 - era) * torch.softmax(lgs.squeeze(), dim=0)[1].item()
                if target >= 0.5:
                    pred = 1
                else:
                    pred = 0
                test_pres = np.append(test_pres, pred)
                test_targets = np.append(test_targets, test_data_loader.dataset[idx][1])
            print("ear:{}".format(era))
            cal_result = cal_score(test_pres.astype(int), test_targets.astype(int))
            with open(data_path + 'result_data/era_result.csv', 'a') as result_file:
                result_file.write('{},{},{},{},{}\n'.format(
                    era,
                    round(cal_result.get_acc(), 4),
                    round(cal_result.get_precision(), 4),
                    round(cal_result.get_recall(), 4),
                    round(cal_result.get_f1(), 4),
                ))
