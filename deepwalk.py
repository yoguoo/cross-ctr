from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
# from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import entropy, kurtosis
import time
import gc
import random
from word2vec import reduce_mem_usage


def deepwalk(df, f1, f2):
    L = 16
    # Deepwalk算法，
    print("deepwalk:", f1, f2)
    # 构建图
    dic = {}
    for item in df[[f1, f2]].values:
        try:
            str(int(item[1]))
            str(int(item[0]))
        except:
            continue
        try:
            dic['item_' + str(int(item[1]))].add('user_' + str(int(item[0])))
        except:
            dic['item_' + str(int(item[1]))] = set(['user_' + str(int(item[0]))])
        try:
            dic['user_' + str(int(item[0]))].add('item_' + str(int(item[1])))
        except:
            dic['user_' + str(int(item[0]))] = set(['item_' + str(int(item[1]))])
    dic_cont = {}
    for key in dic:
        dic[key] = list(dic[key])
        dic_cont[key] = len(dic[key])
    print("creating")
    # 构建路径
    path_length = 24
    sentences = []
    length = []
    for key in dic:
        sentence = [key]
        while len(sentence) != path_length:
            key = dic[sentence[-1]][random.randint(0, dic_cont[sentence[-1]] - 1)]
            if len(sentence) >= 2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)
        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences) % 100000 == 0:
            print(len(sentences))
    print(np.mean(length))
    print(len(sentences))
    # 训练Deepwalk模型
    print('training...')
    random.shuffle(sentences)
    model = Word2Vec(sentences, vector_size=L, window=4, min_count=1, sg=1, workers=10, epochs=20)
    print('outputing...')
    # 输出
    values = set(df[f1].values)
    w2v = []
    for v in values:
        try:
            a = [int(v)]
            a.extend(model['user_' + str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df1 = pd.DataFrame(w2v)
    names = [f1]
    for i in range(L):
        names.append(f1 + '_' + f2 + '_' + names[0] + '_deepwalk_embedding_' + str(L) + '_' + str(i))
    out_df1.columns = names
    print(out_df1.head())

    ########################
    values = set(df[f2].values)
    w2v = []
    for v in values:
        try:
            a = [int(v)]
            a.extend(model['item_' + str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df2 = pd.DataFrame(w2v)
    names = [f2]
    for i in range(L):
        names.append(f1 + '_' + f2 + '_' + names[0] + '_deepwalk_emb_' + str(L) + '_' + str(i))
    out_df2.columns = names
    print(out_df2.head())
    return (out_df1, out_df2)
sort_df=pd.read_csv('/root/digix/dataset/train/train_data_ads.csv',nrows=10000)
emb_cols = [
    ['user_id','task_id'],

]
for f1, f2 in emb_cols:
    out_df1, out_df2 = deepwalk(sort_df, f1, f2)


print('ok')