# -*- coding:utf-8 -*-

import pandas as pd

path = 'dataSet//'

result_one = pd.read_csv(path + 'result_b_20180706181411.csv', encoding='utf-8', sep='\t')
result_two = pd.read_csv(path + 'result_b_20180706093641.csv', encoding='utf-8', sep='\t')

# print(result_one[result_one['RST'] > 0.1])
# print(result_two[result_two['RST'] > 0.1])

max_data_one = result_one['RST'].max()
min_data_one = result_one['RST'].min()
max_data_two = result_two['RST'].max()
min_data_two = result_two['RST'].min()

# result_one['RST'] = [(max_data_one - index) / (max_data_one - min_data_one) for index in result_one['RST']]
# result_two['RST'] = [(max_data_two - index) / (max_data_two - min_data_two) for index in result_two['RST']]

# print(result_one)
# print(result_two)

result_one = result_one.append(result_two)
result_one = result_one.reset_index().drop(['index'], axis=1)
print(result_one)
result_one.to_csv(path + 'fusion.csv', encoding='utf-8', sep='\t', index=None)

