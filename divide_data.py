# -*- coding:utf-8 -*-

import sys
import time
import logging
import numpy as np
import numpy as np
import pandas as pd
from gensim import models
from scipy.stats import mode
from gensim.models import word2vec
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.preprocessing import LabelBinarizer


path = "dataSet//"

'''train'''
train_log = pd.read_csv(path + 'train_log.csv', encoding='utf-8', sep='\t')
train_agg = pd.read_csv(path + 'train_agg.csv', encoding='utf-8', sep='\t')
train_flg = pd.read_csv(path + 'train_flg.csv', encoding='utf-8', sep='\t')

'''test'''
test_log = pd.read_csv(path + 'test_log.csv', encoding='utf-8', sep='\t')
test_agg = pd.read_csv(path + 'test_agg.csv', encoding='utf-8', sep='\t')

'''EVT_LBL one-hot feature'''
model_one_hot = LabelBinarizer()
model_one_hot.fit(train_log['EVT_LBL'])


def return_list(group):
    return list(group)


def return_set(group):
    return set(group)


def return_set_len(group):
    return len(set(group))


def calc_continue_day(group):
    '''最大连续天数'''
    group = sorted(list(set(group)))
    if len(group) <= 1:
        return 0
    flag = 0
    continue_day = 0
    for index in range(0, len(group) - 1, 1):
        if group[index + 1] == (group[index] + 1):
            flag += 1
            if continue_day < flag:
                continue_day = flag
        else:
            flag = 0

    return continue_day + 1

