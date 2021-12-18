# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


from data_cleaning import *
from datetime import datetime
from xg_training import *
from gbdt_training import *
from rf_training import *

data_path = '/Users/tengyujia/local-data/ai-smart/'
pd_config()

# 第一步 修复数据
# pre_data(data_path)
# 聚合 fee_detail
# df = read_file(data_path + 'fee_detail.csv')
# fee_df = group_fee_detail(df)
# fee_df.to_csv(data_path + 'fee_group.csv', index=False)
# 第二步 合并训练集
# df = merge_train(data_path)
# 第三步 查看数据质量：统计每列空值、唯一值数量
# df = read_file(data_path + 'df_merge_train.csv')
# data_quality(df)
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

# ------------------------------------------------------------------------------------
train_data = read_file(data_path + 'person_train.csv')
test_data = read_file(data_path + 'person_test.csv')

# xgboost ------------------------------------------------------------------------------------
# booster = xgb_train(train_data, test_data)
# booster.save_model(data_path + 'models/xg.model')

# booster = read_model(data_path + 'models/xg.model')
# xgb_test(test_data, booster)

# gbdt -------------------------------------------------------------------------
gbdt = gbdt_train(train_data)
gbdt_test(gbdt, test_data)
feature_importance = list(gbdt.feature_importances_)

# random forest ------------------------------------
# rf = rf_train(train_data)
# rf_test(rf, test_data)
# feature_importance = list(rf.feature_importances_)

# print features -----------------------------------
print_features(feature_importance, train_data.columns)
print('-----finished ------')
