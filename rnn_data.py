from data_cleaning import *
from datetime import datetime

data_path = '/Users/tengyujia/local-data/ai-smart/'
pd_config()

df = read_file(data_path + 'renamed_column_train.csv')

to_drop_columns = [
    'id',
    'nameOfDischargeDiagnosis',
    'startTimeOfHospitalization',
    'terminationTimeOfHospitalization',
    'applicationProcessingTime',
    'operatingTime'
]
df.drop(labels=to_drop_columns, axis=1, inplace=True)
df['transactionHour'] = df['transactionHour'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
df.fillna(value=0, inplace=True)
#loc(df, 0, 1000000, data_path + '/rnn_train.csv')
#loc(df, 1100000, 1110000, data_path + '/rnn_test.csv')
df.to_csv(data_path + 'rnn_train_v2.csv', index=False)
