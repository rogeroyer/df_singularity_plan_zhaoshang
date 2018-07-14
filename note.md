### **交叉验证**

```
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
train_feature = pd.read_csv(path + 'train_feature.csv', encoding='utf-8', low_memory=False)
# test_feature = pd.read_csv(path + 'test_feature.csv', encoding='utf-8', low_memory=False)
X_train, X_test, y_train, y_test = train_test_split(train_feature.drop(['USRID', 'FLAG'], axis=1), train_feature[['FLAG']], test_size=.2, random_state=88)
clf = lgb.LGBMClassifier(learning_rate=0.03,
                         n_estimators=250,
                         max_depth=5,
                         subsample=0.7,
                         boosting_type='gbdt',
                         num_leaves=31,
                         nthread=4,
                         scale_pos_weight=1,
                         seed=27)

clf.fit(X_train, y_train['FLAG'].ravel())
score = roc_auc_score(y_test['FLAG'].ravel(), clf.predict_proba(X_test)[:, 1])
print(score)
```

### Need to realize

- 针对没有使用app或H5的用户使用agg表的相关属性进行聚类分析提取相关特征
- 特征：(同一个簇的其它样本均值) / (该样本到簇中心的距离)   距离越大所占权重越小
- 是否使用app\web\H5

- 点击一个模块次数统计取最多或最少等等
- 模块的停留时间等等
- 天数跟那一天加权
- 连续点击同一个模块的次数，并且时间不一样

### log表属性分析
```
     index     V2
0 -0.90689  10922
1  1.10266   9078

     index     V4
0  0.28920  18425
1 -3.45785   1575

     index     V5
0 -0.68454  13599
1  1.46083   6401

       index    V16
0   -0.18800  18174
1    1.09256   1250
2    2.37312    341
3    3.65368    112
4    4.93425     45
5    6.21481     26
6    7.49537     18
7    8.77593     11
8   10.05649     10
9   11.33705      5
10  19.02042      2
11  17.73986      2
12  12.61761      2
13  16.45929      1
14  22.86210      1

       index    V22
0    0.15815  12856
1   -0.77954   5823
2    1.09584    926
3    2.03353    225
4    2.97122     73
5    3.90892     40
6    4.84661     18
7    9.53506      7
8    6.72199      6
9    5.78430      5
10   7.65968      3
11  17.03659      3
12  12.34814      2
13  10.47275      2
14  19.84967      2
15  18.91198      2
16  71.42268      1
17  11.41045      1
18   8.59737      1
19  54.54424      1
20  50.79348      1
21  40.47887      1
22  13.28583      1

       index   V26
0   -0.26368  9380
1   -0.77398  4185
2    0.24662  3358
3    0.75692  1307
4    1.26722   712
5    1.77752   405
6    2.28782   222
7    2.79812   133
8    3.30842    86
9    3.81872    57
10   4.83932    36
11   4.32902    35
12   5.85992    21
13   5.34962    16
14   6.37022    10
15   6.88052     6
16   7.39082     6
17  10.96292     5
18  11.47322     3
19  11.98352     3
20   8.92172     3
21   9.94232     3
22   7.90112     2
23  21.16891     1
24  62.50318     1
25  16.06591     1
26  15.04531     1
27   9.43202     1
28  13.51441     1


```

### 数据探查
```
'''0-1 analyse static'''
train_log = train_log.drop_duplicates(['USRID'])[['USRID', 'TCH_TYP']]
train_flg = train_flg.merge(train_log, on='USRID', how='left')
train_flg['label'] = train_flg.apply(lambda index: 0 if np.isnan(index.TCH_TYP) else 1, axis=1)
train_flg = train_flg.drop(['TCH_TYP'], axis=1)
print(train_flg[train_flg['label'] == 1]['FLAG'].value_counts().reset_index())
print(train_flg[train_flg['label'] == 0]['FLAG'].value_counts().reset_index())

   index   FLAG
0      0  36168
1      1   2860
   index   FLAG
0      0  40656
1      1    316

```

>257-922-1523
>
>38-115-117

#### chi-square test 

> 卡方检验就是统计样本的实际观测值与理论推断值之间的偏离程度，实际观测值与理论推断值之间的偏离程度就决定卡方值的大小，卡方值越大，越不符合；卡方值越小，偏差越小，越趋于符合，若两个值完全相等时，卡方值就为0，表明理论值完全符合。
