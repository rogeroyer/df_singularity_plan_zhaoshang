#coding:utf-8

import pandas as pd
import numpy as np

# 读取个人信息
train_agg = pd.read_csv('../data/train_agg.csv',sep='\t')
test_agg = pd.read_csv('../data/test_agg.csv',sep='\t')
agg = pd.concat([train_agg,test_agg],copy=False)

# 日志信息
train_log = pd.read_csv('../data/train_log.csv',sep='\t')
test_log = pd.read_csv('../data/test_log.csv',sep='\t')
log = pd.concat([train_log,test_log],copy=False)

log['EVT_LBL_1'] = log['EVT_LBL'].apply(lambda x:x.split('-')[0])
log['EVT_LBL_2'] = log['EVT_LBL'].apply(lambda x:x.split('-')[1])
log['EVT_LBL_3'] = log['EVT_LBL'].apply(lambda x:x.split('-')[1])

# 用户唯一标识
train_flg = pd.read_csv('../data/train_flg.csv',sep='\t')
test_flg = pd.read_csv('../data/submit_sample.csv',sep='\t')
test_flg['FLAG'] = -1
del test_flg['RST']
flg = pd.concat([train_flg,test_flg],copy=False)

data = pd.merge(agg,flg,on=['USRID'],how='left',copy=False)

import time
# 这个部分将时间转化为秒，之后计算用户下一次的时间差特征
# 这个部分可以发掘的特征其实很多很多很多很多
log['OCC_TIM'] = log['OCC_TIM'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
log = log.sort_values(['USRID','OCC_TIM'])
log['next_time'] = log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)

log = log.groupby(['USRID'],as_index=False)['next_time'].agg({
    'next_time_mean':np.mean,
    'next_time_std':np.std,
    'next_time_min':np.min,
    'next_time_max':np.max
})

data = pd.merge(data,log,on=['USRID'],how='left',copy=False)


from sklearn.model_selection import StratifiedKFold

train = data[data['FLAG']!=-1]
test = data[data['FLAG']==-1]

train_userid = train.pop('USRID')
y = train.pop('FLAG')
col = train.columns
X = train[col].values

test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
test = test[col].values

N = 5
skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=42)

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

xx_cv = []
xx_pre = []

for train_in,test_in in skf.split(X,y):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 32,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    verbose_eval=250,
                    early_stopping_rounds=50)

    # print('Save models...')
    # save models to file
    # gbm.save_model('models.txt')

    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    xx_cv.append(roc_auc_score(y_test,y_pred))
    xx_pre.append(gbm.predict(test, num_iteration=gbm.best_iteration))

s = 0
for i in xx_pre:
    s = s + i

s = s /N

res = pd.DataFrame()
res['USRID'] = list(test_userid.values)
res['RST'] = list(s)

print('xx_cv',np.mean(xx_cv))

import time
time_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
res.to_csv('../submit/%s_%s.csv'%(str(time_date),str(np.mean(xx_cv)).split('.')[1]),index=False,sep='\t')
