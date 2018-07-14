# -*- coding:utf-8 -*-

import time
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

path = "dataSet//"

'''train'''
train_log = pd.read_csv(path + 'train_log.csv', encoding='utf-8', sep='\t')
# train_agg = pd.read_csv(path + 'train_agg.csv', encoding='utf-8', sep='\t')
train_flg = pd.read_csv(path + 'train_flg.csv', encoding='utf-8', sep='\t')

# '''test'''
test_log = pd.read_csv(path + 'test_log.csv', encoding='utf-8', sep='\t')
# test_agg = pd.read_csv(path + 'test_agg.csv', encoding='utf-8', sep='\t')


def return_list(group):
    return list(group)


def return_set(group):
    return set(group)


def return_set_len(group):
    return len(set(group))


'''agg table static'''
# print(test_agg)
# print(test_agg['V1'].value_counts().reset_index())
# print(test_agg['V2'].value_counts().reset_index())
# print(test_agg['V3'].value_counts().reset_index())
# print(test_agg['V4'].value_counts().reset_index())
# print(test_agg['V5'].value_counts().reset_index())
# print(test_agg['V6'].value_counts().reset_index())
# print(test_agg['V7'].value_counts().reset_index())
# print(test_agg['V8'].value_counts().reset_index())
# print(test_agg['V9'].value_counts().reset_index())
# print(test_agg['V10'].value_counts().reset_index())
# print(test_agg['V11'].value_counts().reset_index())
# print(test_agg['V12'].value_counts().reset_index())
# print(test_agg['V13'].value_counts().reset_index())
# print(test_agg['V14'].value_counts().reset_index())
# print(test_agg['V15'].value_counts().reset_index())
# print(test_agg['V16'].value_counts().reset_index())
# print(test_agg['V17'].value_counts().reset_index())
# print(test_agg['V18'].value_counts().reset_index())
# print(test_agg['V19'].value_counts().reset_index())
# print(test_agg['V20'].value_counts().reset_index())
# print(test_agg['V21'].value_counts().reset_index())
# print(test_agg['V22'].value_counts().reset_index())
# print(test_agg['V23'].value_counts().reset_index())
# print(test_agg['V24'].value_counts().reset_index())
# print(test_agg['V25'].value_counts().reset_index())
# print(test_agg['V26'].value_counts().reset_index())
# print(test_agg['V27'].value_counts().reset_index())
# print(test_agg['V28'].value_counts().reset_index())
# print(test_agg['V29'].value_counts().reset_index())
# print(test_agg['V30'].value_counts().reset_index())

# V4 V5
'''day user count'''
# train_log['OCC_TIM'] = [index.split(' ')[0] for index in train_log['OCC_TIM']]
# train_log = train_log.drop_duplicates(['USRID', 'OCC_TIM'])
# train_log = train_log['OCC_TIM'].value_counts().reset_index().sort_values(by=['index'], ascending=True)
#
# print(train_log)
# print(train_log.describe())
# print(train_log.info())
#
# name = list(train_log['index'])
# N = [index for index in range(len(name))]
# y = list(train_log['OCC_TIM'])
# plt.bar(N, y, color='blue')
# plt.xticks(N, name, rotation=60, fontproperties='SimHei')   # , rotation='vertical' #
# plt.show()

'''cross validation'''
# train_feature = pd.read_csv(path + 'train_feature.csv', encoding='utf-8', low_memory=False)
# # test_feature = pd.read_csv(path + 'test_feature.csv', encoding='utf-8', low_memory=False)
# X_train, X_test, y_train, y_test = train_test_split(train_feature.drop(['USRID', 'FLAG'], axis=1), train_feature[['FLAG']], test_size=.2, random_state=88)
# clf = lgb.LGBMClassifier(learning_rate=0.03,
#                          n_estimators=250,
#                          max_depth=5,
#                          subsample=0.7,
#                          boosting_type='gbdt',
#                          num_leaves=31,
#                          nthread=4,
#                          scale_pos_weight=1,
#                          seed=27)
#
# clf.fit(X_train, y_train['FLAG'].ravel())
# score = roc_auc_score(y_test['FLAG'].ravel(), clf.predict_proba(X_test)[:, 1])
# print(score)

# print(train_log.describe())

# '''0-1 analyse static'''
# train_log = train_log.drop_duplicates(['USRID'])[['USRID', 'TCH_TYP']]
# train_flg = train_flg.merge(train_log, on='USRID', how='left')
# train_flg['label'] = train_flg.apply(lambda index: 0 if np.isnan(index.TCH_TYP) else 1, axis=1)
# train_flg = train_flg.drop(['TCH_TYP'], axis=1)
# print(train_flg[train_flg['label'] == 1]['FLAG'].value_counts().reset_index())
# print(train_flg[train_flg['label'] == 0]['FLAG'].value_counts().reset_index())


'''day hour static'''
# train_log = train_log.merge(train_flg, on='USRID', how='left')
# train_log = train_log[train_log['FLAG'] == 0]
#
# # train_log['day'] = train_log.apply(lambda index: int(index.OCC_TIM.split(' ')[0].split('-')[2]), axis=1)      # day #
# train_log['hour'] = train_log.apply(lambda index: int(index.OCC_TIM.split(' ')[1].split(':')[0]), axis=1)      # day #
#
# train_log = train_log.drop_duplicates(['USRID', 'hour'])
# hour_count = train_log['hour'].value_counts().reset_index().sort_values(by=['index'])
# plt.bar(list(hour_count['index']), list(hour_count['hour']), color='blue')
# plt.show()


# train_log = train_log.merge(train_flg, on='USRID', how='left')
# positive_instance = train_log[train_log['FLAG'] == 1]
# navigate_instance = train_log[train_log['FLAG'] == 0]
# train_log['first'] = [index.split('-')[0] for index in train_log['EVT_LBL']]
# train_log['second'] = [index.split('-')[1] for index in train_log['EVT_LBL']]
# train_log['three'] = [index.split('-')[2] for index in train_log['EVT_LBL']]
# print(positive_instance['three'].value_counts().reset_index())
# print(navigate_instance['three'].value_counts().reset_index())

'''desperate EVT_LBL feature'''
# train_log = train_log.merge(train_flg, on='USRID', how='left')
# # log_evt_lbl = pd.pivot_table(train_log, index='USRID', values='EVT_LBL', aggfunc=return_list).reset_index()
# # log_evt_lbl_test = pd.pivot_table(test_log, index='USRID', values='EVT_LBL', aggfunc=return_list).reset_index()
# # log_evt_lbl = log_evt_lbl.append(log_evt_lbl_test)
# # log_evt_lbl['EVT_LBL'].to_csv(path + 'words.txt', encoding='utf-8', index=None, header=None)
# # train_log['first'] = train_log.apply(lambda index: int(index.EVT_LBL.split('-')[0]), axis=1)
# # train_log['second'] = train_log.apply(lambda index: int(index.EVT_LBL.split('-')[1]), axis=1)
# # train_log['three'] = train_log.apply(lambda index: int(index.EVT_LBL.split('-')[2]), axis=1)
# train_log['first'] = train_log.apply(lambda index: int(index.EVT_LBL.split('-')[0]), axis=1)
# positive_instance = train_log[train_log['FLAG'] == 1]
# navigate_instance = train_log[train_log['FLAG'] == 0]
# first_of_all_t = pd.pivot_table(positive_instance, index='USRID', values='first', aggfunc=return_list).reset_index().rename(columns={'first': 'first_of_all'})
# first_of_all_f = pd.pivot_table(navigate_instance, index='USRID', values='first', aggfunc=return_list).reset_index().rename(columns={'first': 'first_of_all'})
# print(first_of_all_t)
# print(first_of_all_f)

'''data analyse'''
# train_log = train_log.merge(train_flg, on='USRID', how='left')
# train_log['hour'] = train_log.apply(lambda index: index.OCC_TIM.split(' ')[1].split(':')[0], axis=1)
# positive_instance = train_log[train_log['FLAG'] == 1]
# navigate_instance = train_log[train_log['FLAG'] == 0]
# # print(train_log['EVT_LBL'].value_counts())
# pos_hour = positive_instance['hour'].value_counts().reset_index()
# nav_hour = navigate_instance['hour'].value_counts().reset_index()
# pos_hour = pos_hour.sort_values(by=['index'], ascending=True)
# nav_hour = nav_hour.sort_values(by=['index'], ascending=True)
# plt.plot(list(pos_hour['index']), list(pos_hour['hour']), color='red')
# plt.plot(list(nav_hour['index']), list(nav_hour['hour']), color='green')
# plt.show()

'''evt_lbl one-hot encoding'''
# # print(train_log['EVT_LBL'].value_counts().reset_index())
# model_one_hot = LabelBinarizer()
# model_one_hot.fit(train_log['EVT_LBL'])
# # feature = models.transform(test_log['EVT_LBL'])
# feature = model_one_hot.transform(['520-1836-3689', '0-231-277', '326-1041-1678', '38-115-117'])
# feature = np.array(feature)
#
# result = []
# for index in range(feature.shape[1]):
#     result.append(np.sum(feature[:, index]))
# # feature = pd.DataFrame(feature)
# print(result)

'''day_set_length'''
# train_log['day'] = train_log.apply(lambda index: int(index.OCC_TIM.split(' ')[0].split('-')[2]), axis=1)
# log_table_day_set = pd.pivot_table(train_log, index='USRID', values='day', aggfunc=return_set_len).reset_index().rename(columns={'day': 'log_table_day_set'})
# log_table_day_set = log_table_day_set.merge(train_flg, on='USRID', how='left')
# true_log_day = log_table_day_set[log_table_day_set['FLAG'] == 1].sort_values(by=['log_table_day_set'], ascending=True)
# false_log_day = log_table_day_set[log_table_day_set['FLAG'] == 0].sort_values(by=['log_table_day_set'], ascending=True)
# print(true_log_day)
# print(false_log_day)

'''111'''
# log_table = pd.pivot_table(train_log, index='USRID', values='TCH_TYP', aggfunc='count').reset_index().rename(columns={'TCH_TYP': 'log_table'})
# log_table = log_table.merge(train_flg, on='USRID', how='left')
# true_log = log_table[log_table['FLAG'] == 1].sort_values(by=['log_table'], ascending=True)
# false_log = log_table[log_table['FLAG'] == 0].sort_values(by=['log_table'], ascending=True)
# print(true_log)
# print(false_log)

'''show train_test user intersection'''
# train = set(train_log['USRID'])
# test = set(test_log['USRID'])
# print(len(train.intersection(test)))

'''first count'''
# train_log['first'] = train_log.apply(lambda index: int(index.EVT_LBL.split('-')[0]), axis=1)
# train_log = train_log.merge(train_flg, on='USRID', how='left')
# true_log = train_log[train_log['FLAG'] == 1]
# false_log = train_log[train_log['FLAG'] == 0]
# print(true_log['first'].value_counts().reset_index().sort_values(by=['first'], ascending=True))
# print(false_log['first'].value_counts().reset_index().sort_values(by=['first'], ascending=True))
# print(list(set(true_log['first'])))

'''用户点击各模块的次数作统计'''
# train_log = pd.pivot_table(train_log, index=['USRID', 'EVT_LBL'], values='OCC_TIM', aggfunc='count').reset_index().rename(columns={'OCC_TIM': 'EVT_LBL_CNT'})
# train_log = pd.pivot_table(train_log, index=['USRID'], values='EVT_LBL_CNT', aggfunc=[np.max, np.min, np.mean, np.median, np.var, mode, np.ptp, np.std]).reset_index()
# train_log.columns = ['USRID', 'max', 'min', 'mean', 'median', 'var', 'mode', 'ptp', 'std']
# train_log['mode'] = train_log['mode'].apply(lambda x: x[0][0])
# train_log['cv'] = train_log['std'] / train_log['mean']         # 变异系数 #
# train_log = train_log.merge(train_flg, on='USRID', how='left')
# true_log = train_log[train_log['FLAG'] == 1]
# print(true_log)

''''''
# train_log = pd.pivot_table(train_log, index='USRID', values='TCH_TYP', aggfunc=return_list).reset_index().rename(columns={'TCH_TYP': 'list'})
# train_log = train_log.merge(train_flg, on='USRID', how='left')
# # true_log = train_log[(train_log['FLAG'] == 1) & (len(train_log['tch_typ_list']) == 2)]
# # false_log = train_log[(train_log['FLAG'] == 0) & (len(train_log['tch_typ_list']) == 2)]
# train_log['len'] = train_log['list'].map(lambda x: len(x))
# train_log['set_len'] = train_log['list'].map(lambda x: len(set(x)))
# true_log = train_log[train_log['FLAG'] == 1]
# false_log = train_log[train_log['FLAG'] == 0]
# true_log.pop('FLAG')
# false_log.pop('FLAG')
# print(true_log)
# print(false_log)


