import pandas as pd
import numpy as np
import random
from  tqdm import tqdm
from word2vec import reduce_mem_usage,emb_adjust,emb_adjust1,embed_test1,embed_test
import gc
import scipy.special as special
from scipy.stats import entropy, kurtosis
#分布迁移
# def adjust(df, key, feature):
#     if key == 'user_id':
#         mean7 = df[df['date'] < 10][feature].mean()
#         std7 = df[df['date'] < 10][feature].std()
#         mean8 = df[(df['date'] >= 10) & (df['coldu'] == 1)][feature].mean()
#         std8 = df[(df['date'] >= 10) & (df['coldu'] == 1)][feature].std()
#         df.loc[(df['date'] >= 10) & (df['coldu'] == 1), feature]= ((df[(df['date'] >= 8) & (df['coldu'] == 1)][feature] - mean8) / std8 * std7 + mean7)
#     return df


def make_fea(df,df2):
    train=df[df.istest==0]
    test=df[df.istest==1]
    #统计测试集冷启动用户
    uid_list = train.user_id.unique()
    test['coldu'] = test['user_id'].apply(lambda x: 1 if x not in uid_list else 0)


    #count特征
    print('制作count特征')
    cate_cols = ['user_id', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'device_name', 'slot_id',
                 'spread_app_id', 'device_size', 'app_second_class', 'city', 'city_rank',
                 'gender', 'net_type', 'residence', 'emui_dev']
    for f in tqdm(cate_cols):
        tmp = train[f].map(train[f].value_counts())
        count_mean=int(tmp.mean())
        if tmp.var() > 1:
            train[f + '_count'] = tmp
            tmp=train.groupby(f,as_index=False)[f].agg({f+'_count':'count'})
            test = test.merge(tmp, on=[f], how='left').fillna(count_mean)

    #交叉特征
    key = 'user_id'
    feature_target = ['task_id', 'adv_id', 'adv_prim_id', 'slot_id', 'spread_app_id']
    for target in tqdm(feature_target):
        tmp=train.groupby(key, as_index=False)[target].agg({
        key + '_' + target + '_nunique': 'nunique',
    })
        n_mean=int(tmp.mean()[1])
        train = train.merge(tmp, on='user_id', how='left')
        test = test.merge(tmp, on='user_id', how='left').fillna(n_mean)

        tmp = train.groupby(target, as_index=False)[key].agg({
            target + '_' + key + '_nunique': 'nunique',
        })
        n_mean = int(tmp.mean()[1])
        train = train.merge(tmp, on=target, how='left')
        test = test.merge(tmp, on=target, how='left').fillna(n_mean)


     # 源域nunique特征
    cols = [f for f in df2.columns if f not in ['label', 'istest', 'u_userId']]
    for col in tqdm(cols):
        tmp = df2.groupby(['u_userId'])[col].nunique().reset_index()
        tmp.columns = ['user_id', col + '_feeds_nuni']
        train = train.merge(tmp, on='user_id', how='left')
        test=test.merge(tmp,on='user_id', how='left')


    #三交叉

    #贝叶斯ctr
    train.label_x = train.label_x.astype(int)

    all_data=pd.concat([train,test])
    cate_cols = ['user_id', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'device_name', 'slot_id',
                 'spread_app_id', 'device_size', 'app_second_class', 'city', 'city_rank',
                 'gender', 'net_type', 'residence', 'emui_dev']

    # mean_rate = train['label_x'].astype(int).mean()
    feature_list = cate_cols
    # 前所有天的点击率
    for feat_1 in tqdm(feature_list):
        res = pd.DataFrame()
        for period in [18, 19, 20, 21, 22, 23, 24, 25]:
            # 方便冷启动特征填充均值
            tmp_data = pd.DataFrame(index=all_data[all_data.date == period][feat_1].unique())
            tmp_data.index.name = feat_1
            # 第一天
            if period == 18:
                # 曝光次数
                count = train[train['date'] <= period].groupby(feat_1)['label_x'].agg('count')
                # 点击次数
                label = train[train['date'] <= period].groupby(feat_1)['label_x'].agg('sum')
                # 计算均值用于缺失值填充
                mean_value = (label / count).mean()
                # 更新α，β
                HP = HyperParam(1, 1)
                HP.update_from_data_by_moment(count.values, label.values)
                # 得到贝叶斯平滑后点击率
                ctr = ((label + HP.alpha) / (count + HP.alpha + HP.beta))
                ctr = tmp_data.merge(ctr, left_index=True, right_index=True, how='left')
                # 均值填充
                ctr = ctr.fillna(mean_value).reset_index()

            # 测试集25
            elif period > 24:
                count = train[train['date'] < 25].groupby(feat_1)['label_x'].agg('count')
                label = train[train['date'] < 25].groupby(feat_1)['label_x'].agg('sum')
                mean_value = (label / count).mean()
                HP = HyperParam(1, 1)
                HP.update_from_data_by_moment(count.values, label.values)
                ctr = ((label + HP.alpha) / (count + HP.alpha + HP.beta))
                ctr = tmp_data.merge(ctr, left_index=True, right_index=True, how='left')
                ctr = ctr.fillna(mean_value).reset_index()
            else:
                count = train[train['date'] < period].groupby(feat_1)['label_x'].agg('count')
                label = train[train['date'] < period].groupby(feat_1)['label_x'].agg('sum')
                mean_value = (label / count).mean()
                HP = HyperParam(1, 1)
                HP.update_from_data_by_moment(count.values, label.values)
                ctr = ((label + HP.alpha) / (count + HP.alpha + HP.beta))
                ctr = tmp_data.merge(ctr, left_index=True, right_index=True, how='left')
                ctr = ctr.fillna(mean_value).reset_index()
            ctr['date'] = period
            ctr = ctr.rename(columns={'label_x': feat_1 + 'b_rate'})
            res = res.append(ctr, ignore_index=True)
        all_data = pd.merge(all_data, res, how='left', on=[feat_1, 'date'], sort=False)
        # df[feat_1 + '_rate'] = reduce_s(df[feat_1 + '_rate'].fillna(mean_rate))
        #     df[feat_1 + '_rate'] = df[feat_1 + '_rate'].fillna(mean_rate)
        print(feat_1, ' over')
    reduce_mem_usage(all_data)
    train=all_data[all_data.istest==0]
    test=all_data[all_data.istest==1]
    del all_data
    gc.collect()

    #embedding
    emb_cols = [['user_id', 'task_id']]
    sort_df = train.sort_values('date').reset_index(drop=True)
    for f1, f2 in emb_cols:
        tmp, tmp2 ,model= emb_adjust(sort_df, f1, f2, dim=16)
        train = train.merge(tmp, on=[f1, 'date'], how='left').merge(tmp2, on=f2, how='left').fillna(0)
        tmp=tmp.groupby('user_id',as_index=False).apply(lambda t: t[t.date==t.date.max()]).reset_index(drop=True).drop(columns=('date'))
        test= test.merge(tmp, on=[f1], how='left').merge(tmp2, on=f2, how='left').fillna(0)

    emb_cols = [['user_id', 'ad_click_list_v001']]

    for f1, f2 in emb_cols:
        tmp ,model= emb_adjust1(sort_df, f1, f2, dim=8)
        train = train.merge(tmp, on=[f1, 'date'], how='left').fillna(0)
        tmp = tmp.groupby('user_id', as_index=False).apply(lambda t: t[t.date == t.date.max()]).reset_index(
            drop=True).drop(columns=('date'))
        test = test.merge(tmp, on=[f1], how='left').fillna(0)

    cols = ['ad_click_list_v002','ad_click_list_v003', 'u_newsCatInterestsST', "u_click_ca2_news",
            'u_newsCatInterests']
    for col in cols:

        emb_cols = [['user_id', col]]
        for f1, f2 in emb_cols:
            tmp ,model= emb_adjust1(sort_df, f1, f2, dim=8)
            train= train.merge(tmp, on=[f1, 'date'], how='left').fillna(0)
            tmp = tmp.groupby('user_id', as_index=False).apply(lambda t: t[t.date == t.date.max()]).reset_index(
                drop=True).drop(columns=('date'))
            test= test.merge(tmp, on=[f1], how='left').fillna(0)



    #range特征
    #hash编码


    return pd.concat([train,test])

class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            # imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    # 平滑方式1
    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        '''tries ： 展示次数
           success : 点击次数
           iter_num : 迭代次数
           epsilon : 精度
        '''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            # 当迭代稳定时，停止迭代
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        # digamma 指伽马函数，是阶乘在实数与复数域的扩展
        sumfenzialpha = special.digamma(success + alpha) - special.digamma(alpha)
        print(sumfenzialpha)
        # for i in range(len(tries)):
        #     sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
        #     sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
        #     sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))
        return alpha * (sumfenzialpha / sumfenmu), beta * (sumfenzibeta / sumfenmu)

    # 平滑方式2
    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        # 求均值和方差
        mean, var = self.__compute_moment(tries, success)
        # print 'mean and variance: ', mean, var
        # self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean + 0.000001) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)
        # self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)

    def __compute_moment(self, tries, success):
        # 求均值和方差
        '''moment estimation'''
        ctr_list = []
        # var = 0.0
        mean = (success / tries).mean()
        if len(tries) == 1:
            var = 0
        else:
            var = (success / tries).var()
        # for i in range(len(tries)):
        #     ctr_list.append(float(success[i])/tries[i])
        # mean = sum(ctr_list)/len(ctr_list)
        # for ctr in ctr_list:
        #     var += pow(ctr-mean, 2)
        return mean, var

def make_3_nu(train):
    cross_cols = [ 'city','device_name','task_id', 'creat_type_cd',
       'adv_prim_id','spread_app_id','slot_id']
    cross_group_cols = []
    for ind in range(len(cross_cols)):
        for indj in range(ind+1,len(cross_cols)):
            cross_group_cols.append([cross_cols[ind], cross_cols[indj]])
    for f in tqdm(cross_group_cols):
        for col in cross_cols:
            if col in f:
                continue

            train = train.merge(train[f+[col]].groupby(f, as_index=False)[col].agg({
                'cross_{}_{}_nunique'.format(f, col): 'nunique',
                # 熵
                'cross_{}_{}_ent'.format(f, col): lambda x: entropy(x.value_counts() / x.shape[0])
            }), on=f, how='left')

            count_three = ['cross_{}_{}_{}_count'.format(f[0], f[1], col), 'cross_{}_{}_{}_count'.format(f[0], col, f[1]),
                           'cross_{}_{}_{}_count'.format(
                               f[1], f[0], col), 'cross_{}_{}_{}_count'.format(f[1], col, f[0]),
                           'cross_{}_{}_{}_count'.format(
                               col, f[1], f[0]), 'cross_{}_{}_{}_count'.format(col, f[0], f[1])
                           ]
            flag = True
            for cc in count_three:
                if cc in train.columns.values:
                    flag = False

            if flag:
                train = train.merge(train[f+[col, 'user_id']].groupby(f+[col], as_index=False)['user_id'].agg({
                    'cross_{}_{}_{}_count'.format(f[0], f[1], col): 'count'  # 共现次数
                }), on=f+[col], how='left')

    #         if 'cross_{}_{}_count_ratio'.format(col, f) not in df.columns.values:
    #             df['cross_{}_{}_count_ratio'.format(col, f)] = df['cross_{}_{}_count'.format(f, col)] / df[f + '_count'] # 比例偏好
            for cc in count_three:
                if cc in train.columns.values:
                    countfeat = cc

            if 'cross_{}_{}_{}_count_ratio'.format(f[0], f[1], col) not in train.columns.values and \
                    'cross_{}_{}_{}_count_ratio'.format(f[1], f[0], col) not in train.columns.values:

                train['cross_{}_{}_{}_count_ratio'.format(
                    f[0], f[1], col)] = train[countfeat] / train[col + '_count']  # 比例偏好

    return train

def mke_fea_feeds(df):
    train = df[df.istest1 == 0]
    test = df[df.istest1 == 1]
    print('制作count特征')
    cate_cols = ['u_userId', 'i_docId', 'i_s_sourceId', 'i_regionEntity',
                 'i_cat', 'i_entities', 'i_dislikeTimes', 'i_upTimes', 'i_dtype', 'e_ch',
                 'e_m', 'e_po', 'e_pl', 'e_rn', ]
    for f in tqdm(cate_cols):
        tmp = train[f].map(train[f].value_counts())
        count_mean = int(tmp.mean())
        if tmp.var() > 1:
            train[f + '_count'] = tmp
            tmp = train.groupby(f, as_index=False)[f].agg({f + '_count': 'count'})
            test = test.merge(tmp, on=[f], how='left').fillna(count_mean)
    print('制作交叉特征')
    key = 'u_userId'
    feature_target = ['i_docId', 'i_s_sourceId', 'i_regionEntity',
                      'i_cat', 'i_dislikeTimes', 'i_upTimes', 'i_dtype', 'e_ch',
                      'e_m', 'e_po', 'e_pl', 'e_rn']
    for target in tqdm(feature_target):
        tmp = train.groupby(key, as_index=False)[target].agg({
            key + '_' + target + '_nunique': 'nunique',
        })
        n_mean = int(tmp.mean()[1])
        train = train.merge(tmp, on=key, how='left')
        test = test.merge(tmp, on=key, how='left').fillna(n_mean)

        tmp = train.groupby(target, as_index=False)[key].agg({
            target + '_' + key + '_nunique': 'nunique',
        })
        n_mean = int(tmp.mean()[1])
        train = train.merge(tmp, on=target, how='left')
        test = test.merge(tmp, on=target, how='left').fillna(n_mean)

    print("制作ctr特征")

    label = train.label
    label = label.apply(lambda x: 0 if x == -1 else 1)
    train.label = label

    cate_cols = ['u_userId', 'i_docId', 'i_s_sourceId', 'i_regionEntity',
                 'i_cat', 'i_entities', 'i_dislikeTimes', 'i_upTimes', 'i_dtype', 'e_ch',
                 'e_m', 'e_po', 'e_pl', 'e_rn']

    train.label = train.label.astype(int)
    # mean_rate = train['label_x'].astype(int).mean()
    feature_list = cate_cols
    data_feeds=pd.concat((train,test))
    # 前所有天的点击率
    for feat_1 in tqdm(feature_list):
        res = pd.DataFrame()
        for period in [18, 19, 20, 21, 22, 23, 24, 25]:
            # 方便冷启动特征填充均值
            tmp_data = pd.DataFrame(index=data_feeds[data_feeds.date == period][feat_1].unique())
            tmp_data.index.name = feat_1
            # 第一天
            if period == 18:
                # 曝光次数
                count = train[train['date'] <= period].groupby(feat_1)['label'].agg('count')
                # 点击次数
                label = train[train['date'] <= period].groupby(feat_1)['label'].agg('sum')
                # 计算均值用于缺失值填充
                mean_value = (label / count).mean()
                # 更新α，β
                HP = HyperParam(1, 1)
                HP.update_from_data_by_moment(count.values, label.values)
                # 得到贝叶斯平滑后点击率
                ctr = ((label + HP.alpha) / (count + HP.alpha + HP.beta))
                ctr = tmp_data.merge(ctr, left_index=True, right_index=True, how='left')
                # 均值填充
                ctr = ctr.fillna(mean_value).reset_index()

            # 测试集25
            elif period > 24:
                count = train[train['date'] < 25].groupby(feat_1)['label'].agg('count')
                label = train[train['date'] < 25].groupby(feat_1)['label'].agg('sum')
                mean_value = (label / count).mean()
                HP = HyperParam(1, 1)
                HP.update_from_data_by_moment(count.values, label.values)
                ctr = ((label + HP.alpha) / (count + HP.alpha + HP.beta))
                ctr = tmp_data.merge(ctr, left_index=True, right_index=True, how='left')
                ctr = ctr.fillna(mean_value).reset_index()
            else:
                count = train[train['date'] < period].groupby(feat_1)['label'].agg('count')
                label = train[train['date'] < period].groupby(feat_1)['label'].agg('sum')
                mean_value = (label / count).mean()
                HP = HyperParam(1, 1)
                HP.update_from_data_by_moment(count.values, label.values)
                ctr = ((label + HP.alpha) / (count + HP.alpha + HP.beta))
                ctr = tmp_data.merge(ctr, left_index=True, right_index=True, how='left')
                ctr = ctr.fillna(mean_value).reset_index()
            ctr['date'] = period
            ctr = ctr.rename(columns={'label': feat_1 + 'b_rate'})
            res = res.append(ctr, ignore_index=True)
        data_feeds = pd.merge(data_feeds, res, how='left', on=[feat_1, 'date'], sort=False)
        # df[feat_1 + '_rate'] = reduce_s(df[feat_1 + '_rate'].fillna(mean_rate))
        #     df[feat_1 + '_rate'] = df[feat_1 + '_rate'].fillna(mean_rate)
        print(feat_1, ' over')


    print("embedding特征")



    return data_feeds
