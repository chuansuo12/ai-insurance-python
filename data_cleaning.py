import pandas as pd
from pandas.core.frame import DataFrame
from datetime import datetime


def pre_data(data_path=str):
    data = pd.read_csv(data_path + 'temp.csv', low_memory=False)
    column = data['个人编码']  # 取个人编码列
    column = column.drop_duplicates()  # 把个人编码去重
    column.to_csv(data_path + 'temp.csv')  # 处理问题数据


def data_quality(data=DataFrame):
    print('shape：{}'.format(data.shape))
    # 统计空值
    null_sum = data.isnull().sum()
    print('每列的空值: \n{}'.format(null_sum.sort_values()))
    # 统一每列唯一值的数量
    columns = []
    for cln in data.columns:
        columns.append({'name': cln, 'count': data[cln].dropna().nunique()})
    columns.sort(key=lambda x: x['count'])
    print('column unique value:')
    for cln in columns:
        print('{}     {}'.format(cln['name'], cln['count']))


# data.drop(labels={'农民工医疗救助计算金额', '一次性医用材料拒付金额'}, axis=1)  # 删除是空值的列
# data.fillna()


def read_file(file_path=str):
    data = pd.read_csv(file_path, low_memory=False)
    return data


def pd_config():
    pd.set_option('display.max_columns', 160)
    pd.set_option('display.max_rows', None)


def merge_csv(data_path=str):
    train = pd.read_csv(data_path + 'df_train.csv', low_memory=False)
    fee = pd.read_csv(data_path + 'fee_detail.csv', low_memory=False)
    id = pd.read_csv(data_path + 'df_id_train.csv', low_memory=False)
    train = train.merge(fee, on='顺序号')
    train = train.merge(id, on='个人编码')
    train.to_csv(data_path + 'df_merge_train.csv')


def merge_test_csv(data_path=str):
    train = pd.read_csv(data_path + 'df_test.csv', low_memory=False)
    fee = pd.read_csv(data_path + 'fee_detail.csv', low_memory=False)
    id = pd.read_csv(data_path + 'df_id_test.csv', low_memory=False)
    train = train.merge(fee, on='顺序号')
    train = train.merge(id, on='个人编码')
    train.to_csv(data_path + 'df_merge_test.csv')


def merge_train(data_path=str):
    train = pd.read_csv(data_path + 'df_train.csv', low_memory=False)
    id = pd.read_csv(data_path + 'df_id_train.csv', low_memory=False)
    fee = pd.read_csv(data_path + 'fee_group.csv', low_memory=False)
    train = train.merge(id, on='个人编码')
    train = train.merge(fee, on='顺序号')
    return train


def drop_columns(data=DataFrame):
    # 删除空值的列 和 唯一值的列
    to_drop_columns = [
        '农民工医疗救助计算金额',
        '一次性医用材料拒付金额',
        '拒付原因',
        '拒付原因编码',
        '药品费拒付金额',
        '检查费拒付金额',
        '治疗费拒付金额',
        '手术费拒付金额',
        '床位费拒付金额',
        '医用材料费拒付金额',
        '输全血申报金额',
        '成分输血自费金额',
        '成分输血拒付金额',
        '其它拒付金额',
        '一次性医用材料自费金额',
        '输全血按比例自负金额',
        '统筹拒付金额',
        '双笔退费标识',
        '住院天数',
        '非典补助补助金额',
        '家床起付线剩余',
        '手术费自费金额',
        '最高限额以上金额'
    ]
    data.drop(labels=to_drop_columns, axis=1)


def trans_date(data=DataFrame):
    data.apply(func=to_date(), axis=1, columns=['住院开始时间', '住院终止时间', '申报受理时间', '操作时间'])


def to_date(source):
    if source is None:
        return
    return datetime.strptime(source, '%d-%m月 -%y').date()


def get_approve_amount(data):
    if not pd.isna(data['本次审批金额']):
        return data['本次审批金额']
    return data['输全血申报金额'] \
           + data['其它申报金额'] \
           + data['成分输血申报金额'] \
           + data['手术费申报金额'] \
           + data['床位费申报金额'] \
           + data['检查费申报金额'] \
           + data['治疗费申报金额'] \
           + data['一次性医用材料申报金额'] \
           + data['药品费申报金额']


def get_allowance_approve(data):
    if not pd.isna(data['补助审批金额']):
        return data['补助审批金额']
    if data['城乡救助补助金额'] == 0:
        return 0
    return data['可用账户报销金额'] + data['城乡救助补助金额']


def fill_na(df=DataFrame):
    df.fillna(value={'一次性医用材料申报金额': 0, '城乡救助补助金额': 0, '公务员医疗补助基金支付金额': 0}, inplace=True)
    df['本次审批金额'] = df.apply(func=get_approve_amount, axis=1)
    df['补助审批金额'] = df.apply(func=get_allowance_approve, axis=1)


def cal_fee_detail(x=DataFrame):
    amount = 0
    total = 0
    for index, row in x.iterrows():
        amount = row.at['数量'] * row.at['单价'] + amount
        total += row.at['数量']
    return DataFrame({'三目条目': [len(x)], '三目金额': [amount], '三目数量': [total]}).iloc[0]


def group_fee_detail(df=DataFrame):
    return df.groupby(by='顺序号', as_index=False).apply(func=cal_fee_detail)


def add_column(df=DataFrame):
    df['发生金额'] = df['其它发生金额'] \
                 + df['药品费发生金额'] \
                 + df['检查费发生金额'] \
                 + df['治疗费发生金额'] \
                 + df['医用材料发生金额'] \
                 + df['床位费发生金额'] \
                 + df['手术费发生金额']
    df['纯自费金额'] = df['发生金额'] - df['本次审批金额']
    df.round({'发生金额': 2, '纯自费金额': 2})


def rename_column(df=DataFrame):
    df.rename(columns={
        '顺序号': 'id',
        '个人编码': 'personId',
        '医院编码': 'hospitalId',
        '药品费发生金额': 'amountOfDrugExpenses',
        '贵重药品发生金额': 'amountOfValuableDrugs',
        '中成药费发生金额': 'amountOfChinesePatentMedicineExpenses',
        '中草药费发生金额': 'amountOfChineseHerbalMedicineFee',
        '药品费自费金额': 'outOfPocketAmountOfDrugCosts',
        '药品费拒付金额': 'drugFeeRefusalAmount',
        '药品费申报金额': 'drugFeeDeclarationAmount',
        '检查费发生金额': 'amountOfInspectionFee',
        '贵重检查费金额': 'valuableInspectionFeeAmount',
        '检查费自费金额': 'selfPayAmountOfInspectionFee',
        '检查费拒付金额': 'inspectionFeeRefusalAmount',
        '检查费申报金额': 'inspectionFeeDeclaredAmount',
        '治疗费发生金额': 'amountOfTreatmentExpenses',
        '治疗费自费金额': 'selfPayAmountOfTreatmentFee',
        '治疗费拒付金额': 'amountOfRefusalToPayForTreatment',
        '治疗费申报金额': 'amountDeclaredForTreatmentFee',
        '手术费发生金额': 'amountOfSurgeryFee',
        '手术费自费金额': 'surgicalFeeSelfPayAmount',
        '手术费拒付金额': 'surgicalFeeRefusalAmount',
        '手术费申报金额': 'surgicalFeeDeclaredAmount',
        '床位费发生金额': 'amountOfBedFee',
        '床位费拒付金额': 'chargesForRefusalToPayForBed',
        '床位费申报金额': 'declaredAmountOfBedFee',
        '医用材料发生金额': 'amountOfMedicalMaterials',
        '高价材料发生金额': 'amountOfHighPricedMaterials',
        '医用材料费自费金额': 'selfPayAmountOfMedicalMaterials',
        '医用材料费拒付金额': 'amountOfRefusalToPayForMedicalMaterials',
        '输全血申报金额': 'amountDeclaredForWholeBloodTransfusion',
        '成分输血自费金额': 'outOfPocketAmountOfBloodComponentTransfusion',
        '成分输血拒付金额': 'componentTransfusionRefusalAmount',
        '成分输血申报金额': 'componentBloodTransfusionDeclaredAmount',
        '其它发生金额': 'otherIncurredAmounts',
        '其它拒付金额': 'otherChargebacks',
        '其它申报金额': 'otherDeclaredAmount',
        '一次性医用材料自费金额': 'oneTimeMedicalMaterialsSelfPayAmount',
        '一次性医用材料拒付金额': 'amountOfOneTimeMedicalMaterialRefusal',
        '一次性医用材料申报金额': 'amountOfOneTimeMedicalMaterialsDeclared',
        '输全血按比例自负金额': 'proportionalSurchargeForWholeBloodTransfusion',
        '起付线标准金额': 'standardAmountOfDeductibleLine',
        '起付标准以上自负比例金额': 'amountOfSelfSufficiencyAboveTheThreshold',
        '医疗救助个人按比例负担金额': 'proportionateAmountOfMedicalAssistancePaidByIndividuals',
        '最高限额以上金额': 'amountAboveTheMaximumLimit',
        '统筹拒付金额': 'coordinateTheChargebackAmount',
        '基本医疗保险统筹基金支付金额': 'amountPaidByTheBasicMedicalInsurancePoolingFund',
        '交易时间': 'transactionHour',
        '农民工医疗救助计算金额': 'calculatedAmountOfMedicalAssistanceForMigrantWorkers',
        '公务员医疗补助基金支付金额': 'theAmountPaidByTheMedicalAssistanceFundForCivilServants',
        '城乡救助补助金额': 'urbanAndRuralAssistanceSubsidyAmount',
        '可用账户报销金额': 'availableAccountReimbursementAmount',
        '基本医疗保险个人账户支付金额': 'basicMedicalInsurancePersonalAccountPaymentAmount',
        '非账户支付金额': 'nonAccountPaymentAmount',
        '双笔退费标识': 'doubleRefundSign',
        '住院开始时间': 'startTimeOfHospitalization',
        '住院终止时间': 'terminationTimeOfHospitalization',
        '住院天数': 'theNumberOfDaysInHospital',
        '申报受理时间': 'applicationProcessingTime',
        '出院诊断病种名称': 'nameOfDischargeDiagnosis',
        '本次审批金额': 'amountOfThisApproval',
        '补助审批金额': 'subsidyApprovalAmount',
        '医疗救助医院申请': 'medicalAidHospitalApplication',
        '残疾军人医疗补助基金支付金额': 'disabledMilitaryPersonnelMedicalAssistanceFundPaymentAmountunt',
        '民政救助补助金额': 'civilAffairsAssistanceSubsidyAmount',
        '城乡优抚补助金额': 'urbanAndRuralPreferentialCareSubsidyAmount',
        '非典补助补助金额': 'sarsSubsidyAmount',
        '家床起付线剩余': 'homeBedMinimumPaymentLineRemaining',
        '操作时间': 'operatingTime',
        'class': 'classTarget',
        '三目条目': 'trinocularEntry',
        '三目金额': 'trinocularAmount',
        '三目数量': 'numberOfTrinoculars',
        '发生金额': 'amountIncurred',
        '纯自费金额': 'pureOutOfPocketAmount'
    }, inplace=True)
