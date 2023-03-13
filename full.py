import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
from tqdm import *
#----------------核心模型----------------

from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
#----------------交叉验证----------------
from sklearn.model_selection import StratifiedKFold, KFold
#----------------评估指标----------------
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
#----------------忽略报警----------------
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer
from word2vec import emb_adjust,reduce_mem_usage,emb_adjust1,embed_test,embed_test1
import lightgbm as lgb
from fea import make_fea,mke_fea_feeds

############################载入数据######################################

warnings.filterwarnings('ignore')
nrows=None
# 读取训练数据和测试数据
train_data_ads = pd.read_csv('dataset/train/train_data_ads.csv',nrows=nrows)
train_data_feeds = pd.read_csv('dataset/train/train_data_feeds.csv',nrows=nrows)

test_data_ads = pd.read_csv('dataset/test/test_data_ads.csv',nrows=nrows)
test_data_feeds = pd.read_csv('dataset/test/test_data_feeds.csv',nrows=nrows)

# 合并数据
# 合并数据
train_data_ads['istest'] = 0
test_data_ads['istest'] = 1
data_ads = pd.concat([train_data_ads, test_data_ads], axis=0, ignore_index=True)

train_data_feeds['istest1'] = 0
test_data_feeds['istest1'] = 1
data_feeds = pd.concat([train_data_feeds, test_data_feeds], axis=0, ignore_index=True)

# id_list=[  'i_docId',
#     'i_s_sourceId',
#     'i_regionEntity',
#     'i_cat',
#     'i_entities',]
# for feat in id_list:
#     encoder = LabelEncoder().fit(data_feeds[feat])
#     data_feeds[feat] =  encoder.transform(data_feeds[feat])

data_ads = reduce_mem_usage(data_ads)
data_feeds = reduce_mem_usage(data_feeds)
print("载入数据完成")
#按user_id日期拼接源域目标域数据集


#第几天
data_ads['date']=data_ads['pt_d'].astype(str).str[6:8].astype(int)
data_feeds['date']=data_feeds['e_et'].astype(str).str[6:8].astype(int)
#合并数据
feed_group=data_feeds.groupby(['u_userId','date']).sample(1)
data_ads=pd.merge(data_ads,feed_group,left_on=['user_id','date'],right_on=['u_userId','date'],how='left')
#天数匹配不了用同一id任1天补充
# na_list=data_ads[data_ads.u_userId.isna()].index.tolist()
# df=pd.merge(data_ads.iloc[na_list,:38],data_feeds.groupby(['u_userId']).sample(1),left_on=['user_id'],right_on=['u_userId'],how="left")
# df.rename(columns={"date_x":"date"},inplace=True)
# df=df.drop("date_y",axis=1)
# data_ads.iloc[na_list,38:]=df.iloc[:,38:]
#
# del df
# gc.collect()

#制作特征
print('开始制作特征')
data_ads=make_fea(data_ads,data_feeds)

#变长
dim=8
varlen_list=['ad_click_list_v001','ad_click_list_v002','ad_click_list_v003',
                   'u_newsCatInterestsST',
                   "u_click_ca2_news",
                   'u_newsCatInterests',
                   'u_newsCatDislike',
                    'i_entities',
                  'ad_close_list_v001','ad_close_list_v002','ad_close_list_v003']
for col in tqdm(varlen_list):
    data_ads[col]=list(map(lambda x:str(x).split('^'),data_ads[col]))
    data_ads[col] = list(map(str, data_ads[col]))
    tv = HashingVectorizer(n_features=dim)
    outputs = tv.fit_transform(data_ads[col])

    for i in range(dim):
        data_ads['{}_emb_{}'.format(col, i)]=outputs.toarray()[:,i]
data_ads = reduce_mem_usage(data_ads)
data_ads=data_ads.drop(varlen_list,axis=1)

cols = [f for f in data_feeds.columns if f not in ['istest','u_userId','u_newsCatInterests','u_newsCatDislike','u_newsCatInterestsST','u_click_ca2_news','i_docId','i_s_sourceId','i_entities']]
for col in tqdm(cols):
    tmp = data_feeds.groupby(['u_userId'])[col].mean().reset_index()
    tmp.columns = ['user_id', col+'_feeds_mean']
    data_ads = data_ads.merge(tmp, on='user_id', how='left')
data_ads = reduce_mem_usage(data_ads)


# data_ads = data_ads.sort_values(['pt_d'])
# for gap in [1, 2, 3]:
#     for fea in [
#
#         'task_id',
#         'adv_id',
#         'creat_type_cd',
#         'adv_prim_id',
#         'inter_type_cd',
#         'spread_app_id'
#         'slot_id',
#         'hispace_app_tags',
#         'app_second_class',
#     ]:
#         data_ads[fea + 'diff' + str(gap)] = pd.DataFrame(data_ads.groupby([fea])['pt_d'].shift(gap))
#         data_ads[fea + 'diff' + str(gap)] = data_ads['pt_d'] - data_ads[fea + 'diff' + str(gap)]
#
# for gap in[1,2,3]:
#
#         data_ads['diff'+str(gap)] =data_ads['pt_d'].shift(gap)
#         data_ads['diff'+str(gap)]=data_ads['pt_d']-data_ads['diff'+str(gap)]

# for fea in tqdm(['user_id', 'task_id', 'adv_id', 'adv_prim_id', 'spread_app_id']):
#     tmp=data_ads.groupby([fea, 'date'],as_index=False)['e_et'].rank(method='dense',ascending=True)
#     tmp.rename(columns={'e_et':fea+'daterange'},inplace=True)
#     data_ads=pd.concat([data_ads,tmp],axis=1)
#     tmp=data_ads.groupby([fea],as_index=False)['e_et'].rank(method='dense',ascending=True)
#     tmp.rename(columns={'e_et':fea+'range'},inplace=True)
#     data_ads=pd.concat([data_ads,tmp],axis=1)

# 压缩使用内存
data_ads = reduce_mem_usage(data_ads)
#训练
#
#
#
#
log_id_list = data_ads[data_ads.istest == 1]['log_id_x']
drop_list = ['log_id_x', 'e_et', 'coldu', 'u_userId','log_id_y','label_y','pro','cilLabel' ]
cols = [f for f in data_ads.columns if f not in drop_list]
data_ads=data_ads[cols]






cols = [f for f in data_ads.columns if f not in ['istest', 'label_x']]

x_train = data_ads[data_ads.istest == 0][cols]
x_test = data_ads[data_ads.istest == 1][cols]
# import re
# x_train= x_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# x_test=x_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
y_train = data_ads[data_ads.istest == 0]['label_x']
x_train=reduce_mem_usage(x_train)
x_test=reduce_mem_usage(x_test)
del data_ads,data_feeds
gc.collect()


def cv_model(clf, train_x, train_y, test_x, clf_name, seed=2022):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])
    iteration = []
    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} {}************************************'.format(str(i + 1),
                                                                                                 str(seed)))

        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], \
                                     train_y[valid_index]

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'boost_from_average': True,
            'train_metric': True,
            'feature_fraction_seed': 1,
            'learning_rate': 0.05,
            'is_unbalance': False,  # 当训练数据是不平衡的，正负样本相差悬殊的时候，可以将这个属性设为true,此时会自动给少的样本赋予更高的权重
            'num_leaves': 256,  # 一般设为少于2^(max_depth)
            'max_depth': -1,  # 最大的树深，设为-1时表示不限制树的深度
            'min_child_samples': 15,  # 每个叶子结点最少包含的样本数量，用于正则化，避免过拟合
            'max_bin': 200,  # 设置连续特征或大量类型的离散特征的bins的数量
            'subsample': 1,  # Subsample ratio of the training instance.
            'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
            'colsample_bytree': 0.5,  # Subsample ratio of columns when constructing each tree.
            'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
            'subsample_for_bin': 200000,  # Number of samples for constructing bin
            'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
            'reg_alpha': 2.99,  # L1 regularization term on weights
            'reg_lambda': 1.9,  # L2 regularization term on weights
            'nthread': 20,
            'verbose': 0,
            #     'device':'gpu'

        }
        weight = trn_x['date'] / trn_x['date'].max()
        lgb_train = lgb.Dataset(trn_x, trn_y, weight=weight)
        weight = val_x['date'] / val_x['date'].max()
        lgb_val = lgb.Dataset(val_x, val_y, weight=weight)
        model = clf.train(params, lgb_train, num_boost_round=3000, valid_sets=(lgb_val), verbose_eval=200,
                          early_stopping_rounds=100,
                          )

        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        test_pred = model.predict(test_x, num_iteration=model.best_iteration)
        iteration.append(model.best_iteration)
        train[valid_index] = val_pred
        test += test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))



        print(cv_scores)
        print(iteration)
        print('Feature names:', model.feature_name())
        print('Feature importances:', list(model.feature_importance()))


    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    print("best_iter:", np.mean(iteration))
    return train, test

gc.collect()
# #
print("开始训练目标域")
cat_train, cat_test = cv_model(lgb, x_train, y_train, x_test, "cat")
print("开始预测")
x_test['pctr'] = np.around(cat_test, 6)
x_test['log_id'] = log_id_list
x_test['scene_id']=1
# x_test[['scene_id','log_id', 'pctr']].to_csv('result/submission.csv', index=False)
sub=x_test[['scene_id','log_id', 'pctr']]
del x_train,x_test,tmp
gc.collect()
print(('开始训练源域'))
nrows=None
train_data_feeds = pd.read_csv('dataset/train/train_data_feeds.csv',nrows=nrows)


test_data_feeds = pd.read_csv('dataset/test/test_data_feeds.csv',nrows=nrows)




train_data_feeds['istest1'] = 0
test_data_feeds['istest1'] = 1
data_feeds = pd.concat([train_data_feeds, test_data_feeds], axis=0, ignore_index=True)
data_feeds['date']=data_feeds['e_et'].astype(str).str[6:8].astype(int)

data_feeds=mke_fea_feeds(data_feeds)
#前向时间差
data_feeds = data_feeds.sort_values(['e_et']).reset_index(drop=True)
for gap in [1, 2, 3]:
    for fea in ['u_userId', 'i_docId', 'i_s_sourceId', 'i_regionEntity','i_cat','e_ch','e_m','e_pl','e_section']:
        data_feeds[fea + 'diff' + str(gap)] = pd.DataFrame(data_feeds.groupby([fea])['e_et'].shift(gap))
        data_feeds[fea + 'diff' + str(gap)] = data_feeds['e_et'] - data_feeds[fea + 'diff' + str(gap)]

for gap in[1,2,3]:

        data_feeds['diff'+str(gap)] =data_feeds['e_et'].shift(gap)
        data_feeds['diff'+str(gap)]=data_feeds['e_et']-data_feeds['diff'+str(gap)]

# data_feeds=data_feeds.drop(id_list,axis=1)
# for fea in tqdm([ 'i_docId', 'i_s_sourceId', 'i_regionEntity',
#        'i_cat','e_ch','e_m','e_pl','e_section']):
#     tmp=data_feeds.groupby([fea, 'date'],as_index=False)['e_et'].rank(method='dense',ascending=True)
#     tmp.rename(columns={'e_et':fea+'daterange'},inplace=True)
#     data_feeds=pd.concat([data_feeds,tmp],axis=1)
#     tmp=data_feeds.groupby([fea],as_index=False)['e_et'].rank(method='dense',ascending=True)
#     tmp.rename(columns={'e_et':fea+'range'},inplace=True)
#     data_feeds=pd.concat([data_feeds,tmp],axis=1)
data_feeds=reduce_mem_usage(data_feeds)
# train=data_feeds[data_feeds.istest1==0]
# test=data_feeds[data_feeds.istest1==1].reset_index(drop=True)
#
# emb_cols = [['u_userId', 'i_docId']]
# sort_df = train.sort_values('date').reset_index(drop=True)
# for f1, f2 in emb_cols:
#     tmp, tmp2, model = emb_adjust(sort_df, f1, f2, dim=16)
#     train = train.merge(tmp, on=[f1, 'date'], how='left').merge(tmp2, on=f2, how='left').fillna(0)
#     tmp, tmp2 = embed_test(test, f1, f2, dim=16, model=model)
#     test = test.merge(tmp, on=[f1], how='left').merge(tmp2, on=f2, how='left').fillna(0)
#
# cols = ['u_newsCatInterests', 'u_newsCatDislike', 'u_newsCatInterestsST',
#         'u_click_ca2_news', 'i_entities']
# for col in cols:
#
#     emb_cols = [['u_userId', col]]
#     for f1, f2 in emb_cols:
#         tmp, model = emb_adjust1(sort_df, f1, f2, dim=16)
#         train = train.merge(tmp, on=[f1, 'date'], how='left').fillna(0)
#         tmp = embed_test1(test, f2, dim=16, model=model)
#         test = pd.concat([test, tmp], axis=1)
#
# data_feeds = pd.concat([train, test],ignore_index=True)
# data_feeds = reduce_mem_usage(data_feeds)


# # embedding
# train = data_feeds[data_feeds.istest1 == 0]
# test = data_feeds[data_feeds.istest1 == 1].reset_index(drop=True)
# # embedding
# emb_cols = [['u_userId', 'i_docId']]
# sort_df = data_feeds.sort_values('date').reset_index(drop=True)
# for f1, f2 in emb_cols:
#     tmp, tmp2, model = emb_adjust(sort_df, f1, f2, dim=8)
#     data_feeds = data_feeds.merge(tmp, on=[f1, 'date'], how='left').merge(tmp2, on=f2, how='left').fillna(0)
#
# cols = ['u_newsCatInterests', 'u_newsCatDislike', 'u_newsCatInterestsST',
#         'u_click_ca2_news', 'i_entities']
# for col in cols:
#
#     emb_cols = [['u_userId', col]]
#     for f1, f2 in emb_cols:
#         tmp, model = emb_adjust1(sort_df, f1, f2, dim=8)
#         data_feeds = data_feeds.merge(tmp, on=[f1, 'date'], how='left').fillna(0)

data_feeds = reduce_mem_usage(data_feeds)

dim=8
id_list=['u_newsCatInterests','u_newsCatDislike','u_newsCatInterestsST','u_click_ca2_news','i_entities']
for col in tqdm(id_list):
    data_feeds[col]=list(map(lambda x:str(x).split('^'),data_feeds[col]))
    data_feeds[col] = list(map(str, data_feeds[col]))
    tv = HashingVectorizer(n_features=dim)
    outputs = tv.fit_transform(data_feeds[col])

    for i in range(dim):
        data_feeds['{}_emb_{}'.format(col, i)]=outputs.toarray()[:,i]

data_feeds=data_feeds.drop(id_list,axis=1)


data_feeds.rename(columns={"istest1":"istest"},inplace=True)
log_id_list = data_feeds[data_feeds.istest == 1]['log_id']
drop_list = ['log_id', 'e_et','pro','cilLabel','label' ]
cols = [f for f in data_feeds.columns if f not in drop_list]
data_ads=data_feeds[cols]
x_train = data_feeds[data_feeds.istest == 0][cols]
x_test = data_feeds[data_feeds.istest == 1][cols]

y_train = data_feeds[data_feeds.istest == 0]['label']

cat_train, cat_test = cv_model(lgb, x_train, y_train, x_test, "cat")
x_test['pctr'] = np.around(cat_test, 6)
x_test['log_id'] = log_id_list
x_test['scene_id']=2

result=x_test[['scene_id','log_id', 'pctr']]

result=pd.concat([sub,result])
print('over')
result.to_csv('result/submission.csv', index=False)