# -*- coding:utf-8 -*-

import random
from divide_data import *
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from extract_features import extract_feature
from extract_features import extract_one_hot_feature
from extract_features import extract_evt_lbl_features
from extract_features import calc_continue_evt_cnt
from extract_features import extract_evt_lbl_cnt_features
from sklearn.model_selection import train_test_split


def train_xgb_module(store_features=False, store_result=False, feature_select=False, num_round=300):
    if store_features is True:
        '''feature'''
        train_feature = extract_feature(train_agg, train_log)
        test_feature = extract_feature(test_agg, test_log)
        print('extract features successfully!')
        # '''word2vec feature'''
        # train_feature = train_feature.merge(extract_evt_lbl_features(train_log), on='USRID', how='left')
        # test_feature = test_feature.merge(extract_evt_lbl_features(test_log), on='USRID', how='left')
        print('extract word2vec features successfully!')
        '''EVT_LBL one hot feature'''
        train_feature = train_feature.merge(extract_one_hot_feature(train_log), on='USRID', how='left')
        test_feature = test_feature.merge(extract_one_hot_feature(test_log), on='USRID', how='left')
        print('extract one hot features successfully!')
        '''EVT_LBL static feature'''
        train_feature = train_feature.merge(extract_evt_lbl_cnt_features(train_log), on='USRID', how='left')
        test_feature = test_feature.merge(extract_evt_lbl_cnt_features(test_log), on='USRID', how='left')
        print('extract EVT_LBL static features successfully!')
        # '''EVT_LBL continue_cnt feature'''
        # train_feature = train_feature.merge(calc_continue_evt_cnt(train_log), on='USRID', how='left')
        # test_feature = test_feature.merge(calc_continue_evt_cnt(test_log), on='USRID', how='left')
        # print('extract EVT_LBL continue_cnt features successfully!')
        '''store'''
        train_feature = train_feature.merge(train_flg, on='USRID', how='left')
        train_feature.to_csv(path + 'train_feature.csv', encoding='utf-8', index=None)
        test_feature.to_csv(path + 'test_feature.csv', encoding='utf-8', index=None)
        print('store features successfully!')
        # '''add cluster features'''
        # train_feature = pd.read_csv(path + 'train_feature.csv', encoding='utf-8', low_memory=False)
        # test_feature = pd.read_csv(path + 'test_feature.csv', encoding='utf-8', low_memory=False)
        # train_cluster = pd.read_csv(path + 'train_cluster.csv', encoding='utf-8', low_memory=False)
        # test_cluster = pd.read_csv(path + 'test_cluster.csv', encoding='utf-8', low_memory=False)
        # train_feature = train_feature.merge(train_cluster, on='USRID', how='left')
        # test_feature = test_feature.merge(test_cluster, on='USRID', how='left')
    else:
        train_feature = pd.read_csv(path + 'train_feature.csv', encoding='utf-8', low_memory=False)
        test_feature = pd.read_csv(path + 'test_feature.csv', encoding='utf-8', low_memory=False)
        # '''cluster relative'''
        # train_feature = pd.read_csv(path + 'train_feature_filled.csv', encoding='utf-8', low_memory=False)
        # test_feature = pd.read_csv(path + 'test_feature_filled.csv', encoding='utf-8', low_memory=False)
        # train_feature = train_feature.drop(['cluster_label', 'center_distance'], axis=1)
        # test_feature = test_feature.drop(['cluster_label', 'center_distance'], axis=1)
        print('read features successfully!')

    '''no log table'''
    # train_feature = train_feature[train_feature['evt_lbl_cnt'].isnull()]
    # # pos_feature = train_feature[train_feature['FLAG'] == 1]
    # # neg_feature = train_feature[train_feature['FLAG'] == 0]
    # # '''instance sample'''
    # # neg_feature = neg_feature.sample(frac=0.098, replace=True, random_state=88)
    # # train_feature = pos_feature.append(neg_feature)
    # # '''shuffle rows'''
    # # index = [i for i in range(train_feature.shape[0])]
    # # random.shuffle(index)
    # # train_feature = train_feature.set_index([index]).sort_index()
    #
    # test_feature = test_feature[test_feature['evt_lbl_cnt'].isnull()]
    # names = ['V' + str(index) for index in range(1, 31, 1)] + ['USRID']
    # train_feature = train_feature[names + ['FLAG']]
    # test_feature = test_feature[names]

    '''have log table'''
    # train_feature = train_feature[train_feature['evt_lbl_cnt'].notnull()]
    # test_feature = test_feature[test_feature['evt_lbl_cnt'].notnull()]
    # train_feature = train_feature.drop(['first_len_rank', 'second_len_rank', 'three_len_rank', 'evt_lbl_cnt_len_rank', 'evt_lbl_cnt_len_reverse', 'first_len_rank_reverse', 'second_len_rank_reverse', 'three_len_rank_reverse'], axis=1)
    # test_feature = test_feature.drop(['first_len_rank', 'second_len_rank', 'three_len_rank', 'evt_lbl_cnt_len_rank', 'evt_lbl_cnt_len_reverse', 'first_len_rank_reverse', 'second_len_rank_reverse', 'three_len_rank_reverse'], axis=1)

    train_feature['word_distance'] = train_feature['word_distance'].map(lambda x: 1 if x >= 1 else 0)
    test_feature['word_distance'] = test_feature['word_distance'].map(lambda x: 1 if x >= 1 else 0)
    # train_feature.pop('word_distance')
    # test_feature.pop('word_distance')
    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'train_metric': True,
        'subsample': 0.8,
        'learning_rate': 0.03,
        'num_leaves': 96,
        'num_threads': 6,
        'max_depth': 5,
        'colsample_bytree': 0.8,
        'lambda_l2': 0.01,
        'verbose': -1,
        # 'feature_fraction': 0.9,
        # 'bagging_fraction': 0.95,
    }
    x_train, x_test, y_train, y_test = train_test_split(train_feature.drop(['USRID', 'FLAG'], axis=1), train_feature[['FLAG']], test_size=.2, random_state=88)

    if feature_select is True:
        # features_name = ['V1', 'V3', 'V6', 'V7', 'V9', 'V10', 'V11', 'V13', 'V15', 'V16', 'V19', 'V22', 'V23', 'V25', 'V27', 'V28', 'V29', 'V30', 'day_set_len', 'tch_typ_set_len', 'tch_typ0', 'tch_typ2', 'tch_typ0_rate', 'tch_typ2_rate', '1', '3', '6', '8', '9', '10', '13', '14', '18', '19', '21', '22', '23', '25', '26', '30', 'days_mean', 'days_min', 'days_max', 'days_var', 'days_median', 'days_day_var', 'days_day_median', 'days_day_skew', 'days_hour_mean', 'days_hour_min', 'days_hour_max', 'days_hour_skew', 'hour_max', 'hour_var', 'hour_skew', 'evt_lbl_cnt_max', 'first_of_max', 'first_of_min', 'first_of_median', 'second_of_max', 'second_of_min', 'second_of_median', 'three_of_median', 'first_max', 'first_min', 'second_min', 'three_max', 'three_median']
        #  features len:68   300:0.87087545758  400:0.87081954925  500:0.870075481655  #

        features_name = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V12', 'V13', 'V16', 'V19', 'V21', 'V23', 'V24', 'V26', 'V28', 'V29', 'evt_lbl_cnt', 'evt_lbl_cnt_every_day', 'every_evt_lbl_cnt', 'tch_typ_set_len', 'tch_typ0', 'tch_typ0_rate', '2', '6', '7', '9', '10', '11', '12', '18', '20', '22', '24', '25', '28', 'days_mean', 'days_min', 'days_var', 'days_skew', 'continue_days', 'days_day_min', 'days_day_median', 'days_day_kurtosis', 'days_hour_max', 'days_hour_var', 'days_hour_kurtosis', 'hour_min', 'hour_max', 'hour_var', 'hour_median', 'evt_lbl_cnt_max', 'second_of_max', 'second_of_min', 'second_of_median', 'three_of_max', 'first_max', 'second_max', 'second_min', 'second_median', 'three_max', 'three_min']

        x_train = x_train[features_name]
        x_test = x_test[features_name]

    validate_label = np.array(y_train['FLAG'], dtype=np.int8)
    val_train = lgb.Dataset(x_train, label=validate_label)
    validate = lgb.train(params=params, train_set=val_train, num_boost_round=num_round)
    score = roc_auc_score(y_test, validate.predict(x_test))
    print('validate auc:', score)

    if store_result is True:

        '''label set'''
        train_label = train_feature[['FLAG']]
        '''pure feature'''
        train_feature = train_feature.drop(['USRID', 'FLAG'], axis=1)
        test_user = test_feature[['USRID']]
        test_feature = test_feature.drop(['USRID'], axis=1)

        if feature_select is True:
            train_feature = train_feature[features_name]
            test_feature = test_feature[features_name]

        train_label = np.array(train_label['FLAG'], dtype=np.int8)
        train = lgb.Dataset(train_feature, label=train_label)
        model_two = lgb.train(params=params, train_set=train, num_boost_round=num_round)
        result = model_two.predict(test_feature)

        pd.set_option('chained', None)  # remove warning #
        test_user['RST'] = [index for index in result]
        print(test_user)
        '''store result'''
        time_string = time.strftime('_%Y%m%d%H%M%S', time.localtime(time.time()))
        file_name = 'result_b' + time_string + '.csv'
        test_user.to_csv(path + file_name, index=None, encoding='utf-8', sep='\t')
        print('result stored successfully!')
    print('program is over!')


if __name__ == '__main__':
    start_time = time.clock()
    train_xgb_module(store_features=False, store_result=False, feature_select=False, num_round=1000)
    end_time = time.clock()
    print('program spend time：', end_time - start_time, ' sec')

