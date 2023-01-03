#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-28 14:42:35
# @Author  : Guosheng Kang (guoshengkang@gmail.com)
# @Link    : https://guoshengkang.github.io
# @Version : $Id$

import os
import numpy as np
import pandas as pd
import pickle
import gensim
from gensim.models.doc2vec import Doc2Vec  # 从gensim导入doc2vec
from torch import nn
TaggededDocument = gensim.models.doc2vec.TaggedDocument
from sentence_transformers import SentenceTransformer, util

def cos_similarity(arr1=None, arr2=None):
    cos_sim = np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
    # cos_sim = 0.5 + 0.5*cos_sim # 归一化[0,1]
    return cos_sim


# 读取
with open('../result/samples_sen.pickle', 'rb') as f:
    samples = pickle.load(f)

with open('../result/desc_sentences.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    All_desc = stored_data["All_desc"]
    Mashups_desc = stored_data['Mashups_desc']
    APIs_desc = stored_data['APIs_desc']
    # print(len(All_desc), len(Mashups_desc), len(APIs_desc)) 19125 6206 12919

with open('../result/sentences_embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    All_embeddings = stored_data['All_embeddings']
    Mashups_embeddings = stored_data['Mashups_embeddings']
    APIs_embeddings = stored_data['APIs_embeddings']
    # print(len(All_embeddings), len(Mashups_embeddings), len(APIs_embeddings))

# headline
columns = []
label_column = ["label"]
API_id_column = ["API_id"]
Mashup_id_column = ["Mashup_id"]
API_sentran_columns = ["I" + str(k) for k in range(1, 51)]
Mashup_sentran_columns = ["I" + str(k) for k in range(51, 101)]
API_Mashup_similarity_column = ["I101"]
API_popularity_column = ["I102"]
API_compatibility_column = ["I103"]
API_category_column = ["C1"]
API_sub_category_column = ["C2"]
Mashup_category_column = ["C3"]
Mashup_sub_category_column = ["C4"]
all_columns = [label_column, API_id_column, Mashup_id_column, API_sentran_columns, Mashup_sentran_columns, API_Mashup_similarity_column,
               API_popularity_column, API_compatibility_column, API_category_column, API_sub_category_column,
               Mashup_category_column, Mashup_sub_category_column]
for temp_columns in all_columns:
    columns.extend(temp_columns)
print("columns:", columns)

row_number = len(samples)
col_number = len(columns)
print("row_number:", row_number)
print("col_number:", col_number)
m = nn.Linear(384, 50)

df = pd.DataFrame(np.zeros((row_number, col_number)), columns=columns)
for index, sample in enumerate(samples):  # (API,Mashup,label)
    df.loc[index, label_column] = str(sample[2])
    df.loc[index, API_id_column] = int(sample[0]["tags_no"])
    df.loc[index, Mashup_id_column] = int(sample[1]["tags_no"])
    API_sentran_values = All_embeddings[sample[0]["tags_no"]]
    df.loc[index, API_sentran_columns] = m(API_sentran_values).detach().numpy()
    Mashup_sentran_values = All_embeddings[sample[1]["tags_no"]]
    df.loc[index, Mashup_sentran_columns] = m(Mashup_sentran_values).detach().numpy()
    cosine_scores = util.cos_sim(API_sentran_values, Mashup_sentran_values)
    df.loc[index, API_Mashup_similarity_column] = cosine_scores[0][0].detach().numpy()
    df.loc[index, API_popularity_column] = sample[0]["pop"]
    df.loc[index, API_compatibility_column] = sample[0]["compatibility"]
    df.loc[index, API_category_column] = sample[0]["primary_category"]
    df.loc[index, API_sub_category_column] = sample[0]["tags"][1]
    df.loc[index, Mashup_category_column] = sample[1]["primary_category"]
    if len(sample[1]["tags"]) > 1:
        df.loc[index, Mashup_sub_category_column] = sample[1]["tags"][1]
    else:
        df.loc[index, Mashup_sub_category_column] = ""


# save df to csv
df.to_csv("../result/input_data_sen_all.csv", index=False, float_format='%.9f')
