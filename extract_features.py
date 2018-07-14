# -*- coding:utf-8 -*-

from divide_data import *
from scipy.stats import mode


def extract_feature(data_set_one, data_set_two):
    # features = data_set_one[['USRID']]
    features = data_set_one.copy()

    # '''agg table'''
    # features = features.drop(['V2', 'V4', 'V5', 'V16', 'V22', 'V26'], axis=1)
    # features['V2_0'] = features.apply(lambda index: 1 if index.V2 < 0 else 0, axis=1)
    # features['V2_1'] = features.apply(lambda index: 1 if index.V2 > 0 else 0, axis=1)
    # features['V4_0'] = features.apply(lambda index: 1 if index.V4 > 0 else 0, axis=1)
    # features['V4_1'] = features.apply(lambda index: 1 if index.V4 < 0 else 0, axis=1)
    # features['V5_0'] = features.apply(lambda index: 1 if index.V5 < 0 else 0, axis=1)
    # features['V5_1'] = features.apply(lambda index: 1 if index.V5 > 0 else 0, axis=1)
    # # print(features)

    '''log table extract feature'''
    data_set_two['day'] = data_set_two.apply(lambda index: int(index.OCC_TIM.split(' ')[0].split('-')[2]), axis=1)      # day #
    data_set_two['hour'] = data_set_two.apply(lambda index: int(index.OCC_TIM.split(' ')[1].split(':')[0]), axis=1)      # day #

    ''''点击模块名称'''
    evt_lbl_cnt = pd.pivot_table(data_set_two, index='USRID', values='EVT_LBL', aggfunc='count').reset_index().rename(columns={'EVT_LBL': 'evt_lbl_cnt'})
    evt_lbl_set_len = pd.pivot_table(data_set_two, index='USRID', values='EVT_LBL', aggfunc=return_set_len).reset_index().rename(columns={'EVT_LBL': 'evt_lbl_set_len'})
    day_set_len = pd.pivot_table(data_set_two, index='USRID', values='day', aggfunc=return_set_len).reset_index().rename(columns={'day': 'day_set_len'})
    features = features.merge(evt_lbl_cnt, on='USRID', how='left')
    features = features.merge(evt_lbl_set_len, on='USRID', how='left')
    features = features.merge(day_set_len, on='USRID', how='left')
    features['evt_lbl_cnt_every_day'] = features.apply(lambda index: index.evt_lbl_cnt / index.day_set_len, axis=1)
    features['evt_lbl_set_len_every_day'] = features.apply(lambda index: index.evt_lbl_set_len / index.day_set_len, axis=1)
    features['every_evt_lbl_cnt'] = features.apply(lambda index: index.evt_lbl_cnt / index.evt_lbl_set_len, axis=1)

    '''事件类型'''
    tch_typ_list = pd.pivot_table(data_set_two, index='USRID', values='TCH_TYP', aggfunc=return_list).reset_index().rename(columns={'TCH_TYP': 'tch_typ_list'})
    tch_typ_list['tch_typ_cnt'] = tch_typ_list.apply(lambda index: len(index.tch_typ_list), axis=1)
    tch_typ_list['tch_typ_set_len'] = tch_typ_list.apply(lambda index: len(set(index.tch_typ_list)), axis=1)
    tch_typ_list['tch_typ0'] = tch_typ_list.apply(lambda index: index.tch_typ_list.count(0), axis=1)
    tch_typ_list['tch_typ2'] = tch_typ_list.apply(lambda index: index.tch_typ_list.count(2), axis=1)
    tch_typ_list['tch_typ_02'] = tch_typ_list.apply(lambda index: 1 if index.tch_typ_set_len == 2 else 0, axis=1)
    tch_typ_list['tch_typ0_rate'] = tch_typ_list.apply(lambda index: index.tch_typ0 / index.tch_typ_cnt, axis=1)
    tch_typ_list['tch_typ2_rate'] = tch_typ_list.apply(lambda index: index.tch_typ2 / index.tch_typ_cnt, axis=1)
    tch_typ_list = tch_typ_list.drop(['tch_typ_list'], axis=1)
    features = features.merge(tch_typ_list, on='USRID', how='left')
    del tch_typ_list

    '''每天触发多少事件'''
    days_evt_cnt = pd.crosstab(data_set_two['USRID'], data_set_two['day']).reset_index()
    days_evt_cnt.columns = ['USRID', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21', 'u22', 'u23', 'u24', 'u25', 'u26', 'u27', 'u28', 'u29', 'u30', 'u31']
    features = features.merge(days_evt_cnt, on='USRID', how='left')
    del days_evt_cnt
    # features['first_seven_days'] = features['u1']*1 + features['u2']*2 + features['u3']*3 + features['u4']*4 + features['u5']*5 + features['u6']*6 + features['u7']*7
    # features['middle_seven_days'] = features['u13']*1 + features['u14']*2 + features['u15']*3 + features['u16']*4 + features['u17']*5 + features['u18']*6 + features['u19']*7
    # features['last_seven_days'] = features['u25']*1 + features['u26']*2 + features['u27']*3 + features['u28']*4 + features['u29']*5 + features['u30']*6 + features['u31']*7

    '''基于时间的统计特征'''
    days_static_feature = pd.pivot_table(data_set_two, index='USRID', values='day', aggfunc=[np.mean, np.min, np.max, np.var, np.median, kurtosis, skew]).reset_index()
    days_static_feature.columns = ['USRID', 'days_mean', 'days_min', 'days_max', 'days_var', 'days_median', 'days_kurtosis', 'days_skew']
    continue_days = pd.pivot_table(data_set_two, index='USRID', values='day', aggfunc=calc_continue_day).reset_index().rename(columns={'day': 'continue_days'})
    features = features.merge(days_static_feature, on='USRID', how='left')
    features = features.merge(continue_days, on='USRID', how='left')
    del days_static_feature

    ''''基于用户每天的时间统计'''
    temp_table = pd.pivot_table(data_set_two, index=['USRID', 'day'], values='TCH_TYP', aggfunc='count').reset_index().rename(columns={'TCH_TYP': 'days_day_static_feature'})
    days_day_static_feature = pd.pivot_table(temp_table, index='USRID', values='days_day_static_feature', aggfunc=[np.mean, np.min, np.max, np.var, np.median, kurtosis, skew]).reset_index()
    days_day_static_feature.columns = ['USRID', 'days_day_mean', 'days_day_min', 'days_day_max', 'days_day_var', 'days_day_median', 'days_day_kurtosis', 'days_day_skew']
    features = features.merge(days_day_static_feature, on='USRID', how='left')
    del days_day_static_feature

    ''''基于用户每天小时的时间统计'''
    temp_table = pd.pivot_table(data_set_two, index=['USRID', 'day', 'hour'], values='TCH_TYP', aggfunc='count').reset_index().rename(columns={'TCH_TYP': 'days_hour_static_feature'})
    days_hour_static_feature = pd.pivot_table(temp_table, index='USRID', values='days_hour_static_feature', aggfunc=[np.mean, np.min, np.max, np.var, np.median, kurtosis, skew]).reset_index()
    days_hour_static_feature.columns = ['USRID', 'days_hour_mean', 'days_hour_min', 'days_hour_max', 'days_hour_var', 'days_hour_median', 'days_hour_kurtosis', 'days_hour_skew']
    features = features.merge(days_hour_static_feature, on='USRID', how='left')
    del days_hour_static_feature

    '''基于用户小时的统计特征'''
    hour_static_feature = pd.pivot_table(data_set_two, index='USRID', values='hour', aggfunc=[np.mean, np.min, np.max, np.var, np.median, kurtosis, skew]).reset_index()
    hour_static_feature.columns = ['USRID', 'hour_mean', 'hour_min', 'hour_max', 'hour_var', 'hour_median', 'hour_kurtosis', 'hour_skew']
    features = features.merge(hour_static_feature, on='USRID', how='left')
    del hour_static_feature

    # '''基于小时的统计特征'''
    # hour_static = pd.pivot_table(data_set_two, index='USRID', values='hour', aggfunc=return_list).reset_index().rename(columns={'hour': 'hour_static'})
    # hour_static['hour_7'] = hour_static.apply(lambda index: index.hour_static.count(7), axis=1)
    # hour_static['hour_8'] = hour_static.apply(lambda index: index.hour_static.count(8), axis=1)
    # hour_static['hour_7_divide_8'] = hour_static.apply(lambda index: index.hour_7 / index.hour_8 if index.hour_8 != 0 else 0, axis=1)
    # hour_static['hour_13_divide_14'] = hour_static.apply(lambda index: index.hour_static.count(13) / index.hour_static.count(14) if index.hour_static.count(14) != 0 else 0, axis=1)
    # hour_static['hour_23_divide_20'] = hour_static.apply(lambda index: index.hour_static.count(23) / index.hour_static.count(20) if index.hour_static.count(20) != 0 else 0, axis=1)
    # features = features.merge(hour_static.drop(['hour_static'], axis=1), on='USRID', how='left')
    # print(features)

    '''时间名称相关特征（拆分提取）'''
    data_set_two['first'] = data_set_two.apply(lambda index: int(index.EVT_LBL.split('-')[0]), axis=1)
    data_set_two['second'] = data_set_two.apply(lambda index: int(index.EVT_LBL.split('-')[1]), axis=1)
    data_set_two['three'] = data_set_two.apply(lambda index: int(index.EVT_LBL.split('-')[2]), axis=1)
    evt_lbl_cnt_all = pd.pivot_table(data_set_two, index='EVT_LBL', values='USRID', aggfunc='count').reset_index().rename(columns={'USRID': 'evt_lbl_cnt_all'})
    first_of_all = pd.pivot_table(data_set_two, index='first', values='USRID', aggfunc='count').reset_index().rename(columns={'USRID': 'first_of_all'})
    second_of_all = pd.pivot_table(data_set_two, index='second', values='USRID', aggfunc='count').reset_index().rename(columns={'USRID': 'second_of_all'})
    three_of_all = pd.pivot_table(data_set_two, index='three', values='USRID', aggfunc='count').reset_index().rename(columns={'USRID': 'three_of_all'})
    data_set_two = data_set_two.merge(evt_lbl_cnt_all, on='EVT_LBL', how='left')
    data_set_two = data_set_two.merge(first_of_all, on='first', how='left')
    data_set_two = data_set_two.merge(second_of_all, on='second', how='left')
    data_set_two = data_set_two.merge(three_of_all, on='three', how='left')
    # print(data_set_two)

    '''evt_lbl first count'''
    # first = [0, 257, 259, 518, 520, 10, 139, 396, 540, 162, 163, 38, 181, 438, 326, 460, 604, 102, 359, 372, 508]
    # first_name = ['evt_lbl_' + str(index) for index in first]
    # evt_lbl_cnt_first_list = pd.pivot_table(data_set_two, index='USRID', values='first', aggfunc=return_list).reset_index().rename(columns={'first': 'evt_lbl_cnt_first_list'})
    # for index in range(len(first_name)):
    #     evt_lbl_cnt_first_list[first_name[index]] = evt_lbl_cnt_first_list.apply(lambda x: x.evt_lbl_cnt_first_list.count(first[index]), axis=1)
    # features = features.merge(evt_lbl_cnt_first_list.drop(['evt_lbl_cnt_first_list'], axis=1), on='USRID', how='left')

    '''EVT_LBL 拆分特征'''
    evt_lbl_cnt_feature = pd.pivot_table(data_set_two, index='USRID', values='evt_lbl_cnt_all', aggfunc=[np.max, np.min, np.median]).reset_index()
    evt_lbl_cnt_feature.columns = ['USRID', 'evt_lbl_cnt_max', 'evt_lbl_cnt_min', 'evt_lbl_cnt_median']
    first_of_feature = pd.pivot_table(data_set_two, index='USRID', values='first_of_all', aggfunc=[np.max, np.min, np.median]).reset_index()
    first_of_feature.columns = ['USRID', 'first_of_max', 'first_of_min', 'first_of_median']
    second_of_feature = pd.pivot_table(data_set_two, index='USRID', values='second_of_all', aggfunc=[np.max, np.min, np.median]).reset_index()
    second_of_feature.columns = ['USRID', 'second_of_max', 'second_of_min', 'second_of_median']
    three_of_feature = pd.pivot_table(data_set_two, index='USRID', values='three_of_all', aggfunc=[np.max, np.min, np.median]).reset_index()
    three_of_feature.columns = ['USRID', 'three_of_max', 'three_of_min', 'three_of_median']
    features = features.merge(evt_lbl_cnt_feature, on='USRID', how='left')
    features = features.merge(first_of_feature, on='USRID', how='left')
    features = features.merge(second_of_feature, on='USRID', how='left')
    features = features.merge(three_of_feature, on='USRID', how='left')

    # evt_lbl_cnt_len = pd.pivot_table(data_set_two, index='USRID', values='EVT_LBL', aggfunc=return_set_len).reset_index().rename(columns={'EVT_LBL': 'evt_lbl_cnt_len'})
    first_feature = pd.pivot_table(data_set_two, index='USRID', values='first', aggfunc=[np.max, np.min, np.median]).reset_index()
    # first_len = pd.pivot_table(data_set_two, index='USRID', values='first', aggfunc=return_set_len).reset_index().rename(columns={'first': 'first_len'})
    first_feature.columns = ['USRID', 'first_max', 'first_min', 'first_median']
    second_feature = pd.pivot_table(data_set_two, index='USRID', values='second', aggfunc=[np.max, np.min, np.median]).reset_index()
    # second_len = pd.pivot_table(data_set_two, index='USRID', values='second', aggfunc=return_set_len).reset_index().rename(columns={'second': 'second_len'})
    second_feature.columns = ['USRID', 'second_max', 'second_min', 'second_median']
    three_feature = pd.pivot_table(data_set_two, index='USRID', values='three', aggfunc=[np.max, np.min, np.median]).reset_index()
    # three_len = pd.pivot_table(data_set_two, index='USRID', values='three', aggfunc=return_set_len).reset_index().rename(columns={'three': 'three_len'})
    three_feature.columns = ['USRID', 'three_max', 'three_min', 'three_median']
    # features = features.merge(first_feature, on='USRID', how='left')
    # features = features.merge(second_feature, on='USRID', how='left')
    # features = features.merge(three_feature, on='USRID', how='left')
    # features = features.merge(evt_lbl_cnt_len, on='USRID', how='left')
    # features = features.merge(first_len, on='USRID', how='left')
    # features = features.merge(second_len, on='USRID', how='left')
    # features = features.merge(three_len, on='USRID', how='left')
    '''rank features'''
    # features['first_len_rank'] = features['first_len'].rank(ascending=True)
    # features['second_len_rank'] = features['second_len'].rank(ascending=True)
    # features['three_len_rank'] = features['three_len'].rank(ascending=True)
    # features['evt_lbl_cnt_len_rank'] = features['evt_lbl_set_len'].rank(ascending=True)
    # features['evt_lbl_cnt_len_reverse'] = features['evt_lbl_set_len'].rank(ascending=False)
    # features['first_len_rank_reverse'] = features['first_len'].rank(ascending=False)
    # features['second_len_rank_reverse'] = features['second_len'].rank(ascending=False)
    # features['three_len_rank_reverse'] = features['three_len'].rank(ascending=False)
    # # print(features)

    '''用户点击各模块的次数作统计'''
    evt_lbl_cnt_two = pd.pivot_table(data_set_two, index=['USRID', 'EVT_LBL'], values='OCC_TIM', aggfunc='count').reset_index().rename(columns={'OCC_TIM': 'evt_lbl_cnt_two'})
    evt_lbl_cnt_two = pd.pivot_table(evt_lbl_cnt_two, index=['USRID'], values='evt_lbl_cnt_two', aggfunc=[np.max, np.min, np.mean, np.median, np.var, mode, np.ptp, np.std]).reset_index()
    evt_lbl_cnt_two.columns = ['USRID', 'evt_lbl_cnt_two_max', 'evt_lbl_cnt_two_min', 'evt_lbl_cnt_two_mean', 'evt_lbl_cnt_two_median', 'evt_lbl_cnt_two_var', 'evt_lbl_cnt_two_mode', 'evt_lbl_cnt_two_ptp', 'evt_lbl_cnt_two_std']
    evt_lbl_cnt_two['evt_lbl_cnt_two_mode'] = evt_lbl_cnt_two['evt_lbl_cnt_two_mode'].apply(lambda x: x[0][0])
    evt_lbl_cnt_two['evt_lbl_cnt_two_cv'] = evt_lbl_cnt_two['evt_lbl_cnt_two_std'] / evt_lbl_cnt_two['evt_lbl_cnt_two_mean']  # 变异系数 #
    features = features.merge(evt_lbl_cnt_two, on='USRID', how='left')
    del evt_lbl_cnt_two

    print(features.shape)
    return features


'''function test'''
# extract_feature(train_agg, train_log)
# extract_feature(test_agg, test_log)


def extract_one_hot_feature(table_log):

    def ont_hot(group):
        feature = model_one_hot.transform(group)
        feature = np.array(feature)
        result = []
        for index in range(feature.shape[1]):
            result.append(np.sum(feature[:, index]))
        return result

    log_evt_lbl = pd.pivot_table(table_log, index='USRID', values='EVT_LBL', aggfunc=return_list).reset_index().rename(columns={'EVT_LBL': 'log_evt_lbl'})
    log_evt_lbl['evt_lbl_category'] = log_evt_lbl.apply(lambda x: ont_hot(x.log_evt_lbl), axis=1)

    vector = ['evt' + str(index) for index in range(len(model_one_hot.classes_))]
    for flag in range(len(vector)):
        log_evt_lbl[vector[flag]] = log_evt_lbl.apply(lambda x: x.evt_lbl_category[flag], axis=1)

    # print(log_evt_lbl)
    log_evt_lbl = log_evt_lbl.drop(['log_evt_lbl', 'evt_lbl_category'], axis=1)
    print(log_evt_lbl.shape)
    return log_evt_lbl


# extract_one_hot_feature(train_log)
# extract_one_hot_feature(test_log)


def extract_evt_lbl_features(table_log):
    vector_size = 200
    # print('vector_size:', vector_size)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(path + "words_zhaoshang.txt")               # 加载语料
    model = word2vec.Word2Vec(sentences, size=vector_size)                       # 训练skip-gram模型，默认window=5

    def calc_mean_distance(group):
        collection1 = []
        collection2 = []
        for index in group:
            # if index not in models.
            try:
                collection1.append(model.similarity(index, "257-922-1523"))
                collection2.append(model.similarity(index, "38-115-117"))
            except:
                continue

        return (np.ptp(collection1) + np.ptp(collection2)) / 2

    log_evt_lbl = pd.pivot_table(table_log, index='USRID', values='EVT_LBL', aggfunc=return_list).reset_index().rename(columns={'EVT_LBL': 'log_evt_lbl'})
    log_evt_lbl['word_distance'] = log_evt_lbl.apply(lambda index: calc_mean_distance(index.log_evt_lbl), axis=1)
    log_evt_lbl = log_evt_lbl.drop(['log_evt_lbl'], axis=1)
    print(log_evt_lbl.shape)
    return log_evt_lbl


# word_distance = extract_evt_lbl_features(train_log)
# word_distance = word_distance.merge(train_flg, on='USRID', how='left')
# print(word_distance[(word_distance['FLAG'] == 1) & (word_distance['word_distance'] >= 1)])
# print(word_distance[(word_distance['FLAG'] == 1) & (word_distance['word_distance'] <= 1)])
# print(word_distance[(word_distance['FLAG'] == 0) & (word_distance['word_distance'] >= 1)])
# print(word_distance[(word_distance['FLAG'] == 0) & (word_distance['word_distance'] <= 1)])
#
# # print(word_distance[word_distance['FLAG'] == 1])
# # print(word_distance[word_distance['FLAG'] == 0])


def extract_evt_lbl_cnt_features(data):
    '''
    Click event 38-115-117 and 257-922-1523 relative static
    :param data: train_log
    :return: data
    '''
    def calc_evt_lbl(group):
        evt_dict = {}
        for index in group:
            if index not in evt_dict.keys():
                evt_dict[index] = 1
            else:
                evt_dict[index] += 1

        # return max(evt_dict.values())
        one = list(group).count('38-115-117')
        two = list(group).count('257-922-1523')
        three = len(group)
        return [one, two, (one * two), (three - one - two)]

    evt_lbl_list = pd.pivot_table(data, index='USRID', values='EVT_LBL', aggfunc=calc_evt_lbl).reset_index().rename(columns={'EVT_LBL': 'evt_lbl_list'})
    evt_lbl_list['one_cmt'] = evt_lbl_list['evt_lbl_list'].map(lambda x: x[0])
    evt_lbl_list['two_cmt'] = evt_lbl_list['evt_lbl_list'].map(lambda x: x[1])
    evt_lbl_list['three_cmt'] = evt_lbl_list['evt_lbl_list'].map(lambda x: x[2])
    evt_lbl_list['four_cmt'] = evt_lbl_list['evt_lbl_list'].map(lambda x: x[3])
    evt_lbl_list.pop('evt_lbl_list')
    # print(evt_lbl_list)
    print(evt_lbl_list.shape)
    return evt_lbl_list


# extract_evt_lbl_cnt_features(train_log)


def calc_continue_evt_cnt(data):
    '''
    :param data: train_log
    :return: data
    '''
    def calc_continue_evt(group):
        group = list(group)
        count, flag = 0, 0
        for index in range(len(group) - 1):
            if group[index] == group[index + 1]:
                if index + 2 == len(group):
                    flag += 1
                    break
                else:
                    if group[index + 1] != group[index + 2]:
                        flag += 1
            else:
                continue
        return flag

    data = data.sort_values(by=['USRID', 'OCC_TIM'], ascending=True)
    data = pd.pivot_table(data, index='USRID', values='EVT_LBL', aggfunc=calc_continue_evt).reset_index().rename(columns={'EVT_LBL': 'continue_evt_cnt'})
    return data


def duplicate_time_different_max_cnt(data):
    '''
    连续点击相同事件并且在不同时间
    '''
    def inner_function(evt_lbl2, occ_tim2):
        count, flag = 0, [0]
        for index in range(len(evt_lbl2) - 1):
            if (evt_lbl2[index] == evt_lbl2[index + 1]) and (occ_tim2[index] != occ_tim2[index + 1]):
                if index + 2 == len(evt_lbl2):
                    count += 1
                    flag.append(count)
                    break
                else:
                    if evt_lbl2[index + 1] != evt_lbl2[index + 2]:
                        count += 1
                        flag.append(count)
                        count = 0
                    else:
                        count += 1
                        if occ_tim2[index + 1] == occ_tim2[index + 2]:
                            flag.append(count)
                            count = 0
            else:
                count = 0

        return max(flag), np.mean(flag), np.var(flag)

    data = data.sort_values(by=['USRID', 'OCC_TIM'], ascending=True)
    evt_lbl = pd.pivot_table(data, index='USRID', values='EVT_LBL', aggfunc=return_list).reset_index().rename(columns={'EVT_LBL': 'evt_lbl'})
    occ_tim = pd.pivot_table(data, index='USRID', values='OCC_TIM', aggfunc=return_list).reset_index().rename(columns={'OCC_TIM': 'occ_tim'})
    evt_lbl = evt_lbl.merge(occ_tim, on='USRID', how='left')
    evt_lbl['duplicate_time_different_max_cnt'] = evt_lbl.apply(lambda index: inner_function(index.evt_lbl, index.occ_tim), axis=1)
    evt_lbl['duplicate_time_different_max_cnt_one'] = evt_lbl['duplicate_time_different_max_cnt'].map(lambda x: x[0])
    evt_lbl['duplicate_time_different_max_cnt_two'] = evt_lbl['duplicate_time_different_max_cnt'].map(lambda x: x[1])
    evt_lbl['duplicate_time_different_max_cnt_three'] = evt_lbl['duplicate_time_different_max_cnt'].map(lambda x: x[2])
    evt_lbl.pop('evt_lbl')
    evt_lbl.pop('occ_tim')
    evt_lbl.pop('duplicate_time_different_max_cnt')
    return evt_lbl


# print(duplicate_time_different_max_cnt(train_log))

