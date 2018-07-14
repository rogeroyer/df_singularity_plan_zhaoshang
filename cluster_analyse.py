# -*- coding:utf-8 -*-

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def calc_distance(origin, new):
    result = 0
    for index in range(len(origin)):
        result += (new[index] - origin[index]) ** 2
    return math.sqrt(result)


path = "dataSet//"

'''cluster analyse'''
# data = pd.read_csv(path + 'train_agg.csv', encoding='utf-8', sep='\t')
# test_data = pd.read_csv(path + 'test_agg.csv', encoding='utf-8', sep='\t')
# user_id = data[['USRID']]
# data = data.drop(['USRID'], axis=1)
# train_flg = pd.read_csv(path + 'train_flg.csv', encoding='utf-8', sep='\t')
# data = data.values
# estimator = KMeans(n_clusters=2, max_iter=1000, verbose=0, algorithm='auto')
# estimator.fit(data)
# lable_pred = estimator.labels_               # 簇标签 #
# centroids = estimator.cluster_centers_       # 簇中心 #
#
# names = ['V' + str(index) for index in range(1, 31, 1)]
# data = pd.DataFrame(data)
# data.columns = names
# data['cluster_label'] = [index for index in lable_pred]
# data['USRID'] = [index for index in user_id['USRID']]
# data = data.merge(train_flg, on='USRID', how='left')
#
# true_distance = centroids[0]
# false_distance = centroids[1]
#
# # print(data[(data['cluster_label'] == 1) & (data['FLAG'] == 1)].shape)
# # print(data[(data['cluster_label'] == 0) & (data['FLAG'] == 0)].shape)
#
# print(data[data['cluster_label'] == 1].shape)
# print(data[data['cluster_label'] == 0].shape)
# # print(data[data['FLAG'] == 1].shape)
# # print(data[data['FLAG'] == 0].shape)
#
# # print(centroids)
#
# train_value = data.drop(['cluster_label', 'USRID', 'FLAG'], axis=1).values
# label = list(data['cluster_label'])
# # print(label)
#
# '''计算训练集样本到簇中心的距离'''
# values = []
# for index in range(len(train_value)):
#     if label[index] == 1:
#         values.append(calc_distance(train_value[index], false_distance))
#     else:
#         values.append(calc_distance(train_value[index], true_distance))
#
# data['center_distance'] = [index for index in values]
# print(data)
#
#
# '''计算测试集聚类结果'''
# predict = estimator.predict(test_data.drop(['USRID'], axis=1).values)
# # print(pd.DataFrame(predict, columns=['cluster_label']))
# test_data['cluster_label'] = [index for index in predict]
# # print(test_data)
# test_value = test_data.drop(['cluster_label', 'USRID'], axis=1).values
#
# '''计算测试集样本到簇中心的距离'''
# test_values = []
# for index in range(len(test_value)):
#     if label[index] == 1:
#         test_values.append(calc_distance(test_value[index], false_distance))
#     else:
#         test_values.append(calc_distance(test_value[index], true_distance))
#
# test_data['center_distance'] = [index for index in test_values]
# print(test_data)
#
# data[['USRID', 'cluster_label', 'center_distance']].to_csv(path + 'train_cluster.csv', encoding='utf-8', index=None)
# test_data[['USRID', 'cluster_label', 'center_distance']].to_csv(path + 'test_cluster.csv', encoding='utf-8', index=None)


train_feature = pd.read_csv(path + 'train_feature.csv', encoding='utf-8', low_memory=False)
test_feature = pd.read_csv(path + 'test_feature.csv', encoding='utf-8', low_memory=False)
train_cluster = pd.read_csv(path + 'train_cluster.csv', encoding='utf-8', low_memory=False)
test_cluster = pd.read_csv(path + 'test_cluster.csv', encoding='utf-8', low_memory=False)
train_feature = train_feature.merge(train_cluster, on='USRID', how='left')
test_feature = test_feature.merge(test_cluster, on='USRID', how='left')

# print(train_feature[train_feature['cluster_label'] == 1])
# print(test_feature[test_feature['first_median'].notnull()]['first_median'].mean())

train_true_value = dict()
train_false_value = dict()
for index in list(train_feature.columns[31:125]):
    train_true_value[index] = train_feature[train_feature['cluster_label'] == 1][index].mean()
    train_false_value[index] = train_feature[train_feature['cluster_label'] == 0][index].mean()

print(train_true_value)
print(train_false_value)


test_true_value = dict()
test_false_value = dict()
for index in list(test_feature.columns[31:125]):                                                           ##  确定要填充的特征  ##
    test_true_value[index] = test_feature[test_feature['cluster_label'] == 1][index].mean()
    test_false_value[index] = test_feature[test_feature['cluster_label'] == 0][index].mean()

print(test_true_value)
print(test_false_value)


def fill_value(cluster_label, center_distance, column_name, index1, name):
    # print(cluster_label, center_distance, column_name, index1, name)
    if name == 'train':
        if column_name > train_set_min:
            return column_name
        else:
            if cluster_label > 0.0:
                return train_true_value[index1] - center_distance    # + - * / #
            else:
                return train_false_value[index1] - center_distance
    else:
        if column_name > test_set_min:
            return column_name
        else:
            if cluster_label > 0.0:
                return test_true_value[index1] - center_distance
            else:
                return test_false_value[index1] - center_distance


train_set_min = train_feature.min().reset_index()[0].min() - 10
test_set_min = test_feature.min().reset_index()[0].min() - 10

for index2 in list(test_feature.columns[31:125]):
    train_feature[index2] = train_feature.apply(lambda x: fill_value(x.cluster_label, x.center_distance, x[index2], index2, 'train'), axis=1)
    print('train_set ', index2, 'is OK!')
    test_feature[index2] = test_feature.apply(lambda y: fill_value(y.cluster_label, y.center_distance, y[index2], index2, 'test'), axis=1)
    print('test_set', index2, ' is OK!')


print(train_feature)
print(test_feature)

train_feature.to_csv(path + 'train_feature_filled.csv', encoding='utf-8', index=None)
test_feature.to_csv(path + 'test_feature_filled.csv', encoding='utf-8', index=None)

