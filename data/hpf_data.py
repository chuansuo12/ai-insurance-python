from data_cleaning import *

pd_config()
data_dir = '/Users/tengyujia/local-data/hpf/'
inpatient = data_dir + 'Train_Inpatientdata-1542865627584.csv'
outpatient = data_dir + 'Train_Outpatientdata-1542865627584.csv'
beneficiary = data_dir + 'Train_Beneficiarydata-1542865627584.csv'


if __name__ == '__main__':
    # 第三步 查看数据质量：统计每列空值、唯一值数量
    df = read_file(beneficiary)
    data_quality(df)
    # 第四步 删除空值、转换日期类型（不做也罢，这几列都用不到）
    # 第五步 填充空值
    # df = read_file(data_path + 'df_merged_train.csv')
    # fill_na(df)
    # df.to_csv(data_path + 'df_train_fill_na.csv', index=False)
    # 第六步 新增列
    # df = read_file(data_path + 'df_train_fill_na.csv')
    # add_column(df)
    # df.to_csv(data_path + 'df_train_fill_na_add_column.csv', index=False)
    # 第七步 重命名列
    # df = read_file(data_path + 'df_train_fill_na_add_column.csv')
    # rename_column(df)
    # print(q2.head(10))
    # df.to_csv(data_path + 'renamed_column_train.csv', index=False)
