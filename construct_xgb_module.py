# -*- coding:utf-8 -*-

import random
from divide_data import *
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from extract_features import extract_feature
from extract_features import extract_one_hot_feature
from extract_features import extract_evt_lbl_features
from extract_features import calc_continue_evt_cnt
from extract_features import duplicate_time_different_max_cnt
from extract_features import extract_evt_lbl_cnt_features
from sklearn.model_selection import train_test_split


def train_xgb_module(store_features=False, store_result=False, feature_select=False, num_round=300):
    if store_features is True:
        '''feature'''
        train_feature = extract_feature(train_agg, train_log)
        test_feature = extract_feature(test_agg, test_log)
        print('extract features successfully!')
        '''word2vec feature'''
        train_feature = train_feature.merge(extract_evt_lbl_features(train_log), on='USRID', how='left')
        test_feature = test_feature.merge(extract_evt_lbl_features(test_log), on='USRID', how='left')
        print('extract word2vec features successfully!')
        '''EVT_LBL one hot feature'''
        train_feature = train_feature.merge(extract_one_hot_feature(train_log), on='USRID', how='left')
        test_feature = test_feature.merge(extract_one_hot_feature(test_log), on='USRID', how='left')
        print('extract one hot features successfully!')
        '''EVT_LBL static feature'''
        train_feature = train_feature.merge(extract_evt_lbl_cnt_features(train_log), on='USRID', how='left')
        test_feature = test_feature.merge(extract_evt_lbl_cnt_features(test_log), on='USRID', how='left')
        print('extract EVT_LBL static features successfully!')
        '''EVT_LBL continue_cnt feature'''
        train_feature = train_feature.merge(calc_continue_evt_cnt(train_log), on='USRID', how='left')
        test_feature = test_feature.merge(calc_continue_evt_cnt(test_log), on='USRID', how='left')
        print('extract EVT_LBL continue_cnt features successfully!')
        '''duplicate_time  feature'''
        train_feature = train_feature.merge(duplicate_time_different_max_cnt(train_log), on='USRID', how='left')
        test_feature = test_feature.merge(duplicate_time_different_max_cnt(test_log), on='USRID', how='left')
        print('extract duplicate_time features successfully!')
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
        'booster': 'gbtree',
        'max_depth': 5,
        'colsample': 0.8,
        'subsample': 0.8,
        'eta': 0.03,
        'silent': 1,
        'objective': 'binary:logistic',      ## binary:logistic ##
        'eval_metric': 'auc',
        'min_child_weight': 5,
        'scale_pos_weight': 1,
        'nthread': 6,
        # 'seed': 4396,
    }

    x_train, x_test, y_train, y_test = train_test_split(train_feature.drop(['USRID', 'FLAG'], axis=1), train_feature[['FLAG']], test_size=.2, random_state=88)

    if feature_select is True:
        # features_name = ['V1', 'V3', 'V6', 'V7', 'V9', 'V10', 'V11', 'V13', 'V15', 'V16', 'V19', 'V22', 'V23', 'V25', 'V27', 'V28', 'V29', 'V30', 'day_set_len', 'tch_typ_set_len', 'tch_typ0', 'tch_typ2', 'tch_typ0_rate', 'tch_typ2_rate', '1', '3', '6', '8', '9', '10', '13', '14', '18', '19', '21', '22', '23', '25', '26', '30', 'days_mean', 'days_min', 'days_max', 'days_var', 'days_median', 'days_day_var', 'days_day_median', 'days_day_skew', 'days_hour_mean', 'days_hour_min', 'days_hour_max', 'days_hour_skew', 'hour_max', 'hour_var', 'hour_skew', 'evt_lbl_cnt_max', 'first_of_max', 'first_of_min', 'first_of_median', 'second_of_max', 'second_of_min', 'second_of_median', 'three_of_median', 'first_max', 'first_min', 'second_min', 'three_max', 'three_median']
        #  features len:68   300:0.87087545758  400:0.87081954925  500:0.870075481655  #

        # features_name = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V12', 'V13', 'V16', 'V19', 'V21', 'V23', 'V24', 'V26', 'V28', 'V29', 'evt_lbl_cnt', 'evt_lbl_cnt_every_day', 'every_evt_lbl_cnt', 'tch_typ_set_len', 'tch_typ0', 'tch_typ0_rate', '2', '6', '7', '9', '10', '11', '12', '18', '20', '22', '24', '25', '28', 'days_mean', 'days_min', 'days_var', 'days_skew', 'continue_days', 'days_day_min', 'days_day_median', 'days_day_kurtosis', 'days_hour_max', 'days_hour_var', 'days_hour_kurtosis', 'hour_min', 'hour_max', 'hour_var', 'hour_median', 'evt_lbl_cnt_max', 'second_of_max', 'second_of_min', 'second_of_median', 'three_of_max', 'first_max', 'second_max', 'second_min', 'second_median', 'three_max', 'three_min']
        features_name = ['V1', 'V3', 'V7', 'V9', 'V10', 'V12', 'V14', 'V16', 'V19', 'V22', 'V23', 'V24', 'V26', 'V28', 'V29', 'V30', 'evt_lbl_cnt', 'evt_lbl_cnt_every_day', 'evt_lbl_set_len_every_day', 'tch_typ_set_len', 'tch_typ0', 'tch_typ2', 'tch_typ_02', 'u1', 'u5', 'u6', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17', 'u19', 'u20', 'u21', 'u23', 'u24', 'u26', 'u29', 'u30', 'days_mean', 'days_min', 'days_var', 'days_kurtosis', 'days_day_min', 'days_hour_mean', 'days_hour_max', 'days_hour_median', 'days_hour_skew', 'hour_max', 'hour_var', 'evt_lbl_cnt_median', 'first_of_min', 'first_of_median', 'second_of_max', 'second_of_median', 'three_of_max', 'three_of_min', 'three_of_median', 'evt_lbl_cnt_two_max', 'evt_lbl_cnt_two_var', 'evt_lbl_cnt_two_mode', 'evt0', 'evt2', 'evt3', 'evt4', 'evt5', 'evt8', 'evt10', 'evt12', 'evt14', 'evt17', 'evt18', 'evt19', 'evt21', 'evt23', 'evt24', 'evt25', 'evt27', 'evt29', 'evt30', 'evt33', 'evt36', 'evt37', 'evt38', 'evt41', 'evt45', 'evt47', 'evt48', 'evt49', 'evt51', 'evt52', 'evt54', 'evt55', 'evt59', 'evt62', 'evt63', 'evt64', 'evt68', 'evt69', 'evt70', 'evt71', 'evt72', 'evt73', 'evt76', 'evt77', 'evt78', 'evt80', 'evt81', 'evt83', 'evt88', 'evt90', 'evt91', 'evt92', 'evt96', 'evt98', 'evt100', 'evt101', 'evt102', 'evt103', 'evt108', 'evt109', 'evt110', 'evt111', 'evt112', 'evt116', 'evt117', 'evt119', 'evt120', 'evt121', 'evt125', 'evt128', 'evt130', 'evt132', 'evt135', 'evt137', 'evt138', 'evt139', 'evt142', 'evt143', 'evt145', 'evt150', 'evt151', 'evt152', 'evt154', 'evt155', 'evt156', 'evt159', 'evt160', 'evt162', 'evt163', 'evt166', 'evt168', 'evt169', 'evt171', 'evt172', 'evt173', 'evt174', 'evt175', 'evt176', 'evt177', 'evt179', 'evt182', 'evt183', 'evt185', 'evt186', 'evt190', 'evt191', 'evt192', 'evt198', 'evt199', 'evt200', 'evt201', 'evt202', 'evt203', 'evt204', 'evt206', 'evt208', 'evt209', 'evt211', 'evt212', 'evt213', 'evt215', 'evt216', 'evt217', 'evt220', 'evt223', 'evt224', 'evt227', 'evt229', 'evt232', 'evt235', 'evt236', 'evt237', 'evt238', 'evt239', 'evt240', 'evt241', 'evt243', 'evt244', 'evt248', 'evt249', 'evt251', 'evt254', 'evt257', 'evt258', 'evt261', 'evt267', 'evt268', 'evt269', 'evt271', 'evt273', 'evt277', 'evt279', 'evt282', 'evt284', 'evt288', 'evt291', 'evt296', 'evt300', 'evt302', 'evt309', 'evt311', 'evt312', 'evt313', 'evt315', 'evt316', 'evt319', 'evt320', 'evt323', 'evt325', 'evt327', 'evt331', 'evt332', 'evt333', 'evt334', 'evt335', 'evt341', 'evt342', 'evt345', 'evt346', 'evt347', 'evt348', 'evt349', 'evt351', 'evt354', 'evt355', 'evt356', 'evt357', 'evt358', 'evt360', 'evt363', 'evt364', 'evt367', 'evt368', 'evt369', 'evt370', 'evt374', 'evt375', 'evt376', 'evt377', 'evt378', 'evt380', 'evt381', 'evt382', 'evt385', 'evt390', 'evt394', 'evt395', 'evt396', 'evt397', 'evt400', 'evt401', 'evt402', 'evt403', 'evt404', 'evt405', 'evt408', 'evt411', 'evt413', 'evt414', 'evt416', 'evt421', 'evt422', 'evt425', 'evt427', 'evt428', 'evt433', 'evt436', 'evt440', 'evt441', 'evt442', 'evt443', 'evt444', 'evt445', 'evt447', 'evt449', 'evt450', 'evt451', 'evt453', 'evt454', 'evt455', 'evt457', 'evt459', 'evt462', 'evt463', 'evt464', 'evt465', 'evt466', 'evt469', 'evt471', 'evt472', 'evt473', 'evt475', 'evt478', 'evt482', 'evt484', 'evt485', 'evt489', 'evt492', 'evt500', 'evt501', 'evt503', 'evt504', 'evt507', 'evt508', 'evt511', 'evt512', 'evt513', 'evt515', 'evt520', 'evt524', 'evt525', 'evt526', 'evt528', 'evt529', 'evt531', 'evt540', 'evt541', 'evt544', 'evt546', 'evt548', 'evt549', 'evt550', 'evt552', 'evt553', 'evt554', 'evt561', 'evt562', 'evt564', 'evt566', 'evt567', 'evt568', 'evt569', 'evt572', 'evt574', 'evt575', 'evt578', 'evt580', 'evt583', 'evt584', 'evt585', 'evt588', 'evt592', 'evt593', 'evt594']

        x_train = x_train[features_name]
        x_test = x_test[features_name]

    val_train = xgb.DMatrix(x_train, label=y_train)
    x_test = xgb.DMatrix(x_test)
    print('I\'m training validate module.')
    clf = xgb.train(params, val_train, num_round)
    score = roc_auc_score(y_test, clf.predict(x_test))
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

        train = xgb.DMatrix(train_feature, label=train_label)
        test_feature = xgb.DMatrix(test_feature)
        print('I\'m training final module.')
        module_two = xgb.train(params, train, num_round)
        result = module_two.predict(test_feature)
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
    train_xgb_module(store_features=False, store_result=True, feature_select=True, num_round=1000)
    end_time = time.clock()
    print('program spend time：', end_time - start_time, ' sec')


'''
no continue_cnt feature and one hot feature and duplicate_time features  online:8647...  724 features (no user_id & flag)
with continue_cnt feature and one hot feature and duplicate_time features  online:8632...  728 features (no user_id & flag)
'''
