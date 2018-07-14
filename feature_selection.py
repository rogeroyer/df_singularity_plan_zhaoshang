# -*- coding:utf-8 -*-

import math
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

path = "dataSet//"

train_feature = pd.read_csv(path + 'train_feature.csv', encoding='utf-8')
# train_feature = train_feature[train_feature['evt594'].notnull()].fillna(0)
# print(train_feature)
train_label = train_feature[['FLAG']]
train_feature = train_feature.drop(['USRID', 'FLAG'], axis=1)

model = RFE(estimator=LGBMClassifier(), n_features_to_select=100).fit_transform(train_feature, train_label['FLAG'].ravel())
print(model)
