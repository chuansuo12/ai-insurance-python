from data_cleaning import *
dir_path = '/Users/tengyujia/local-data/ai-smart/'

source = read_file(dir_path + 'rnn_train_v2.csv')
source['personId'] = source['personId'] - 352120000000000
source = source.astype('float32')
ids = source['personId'].unique()

print(len(ids))
