# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:40:33 2020

@author: Yang Lechuan
"""
import math

R = [[81, 0, 71, 70, 69], [53, 0, 71, 70, 69], [61, 0, 71, 70, 69]]  # 推荐列表
T = [[81],[53],[61]]  # 真实列表


def cal_indicators(rankedlist, testlist):
    HITS_i = 0
    sum_precs = 0
    AP_i = 0
    len_R = 0
    len_T = 0
    MRR_i = 0
    HR_i = 0
    DCG_i = 0
    IDCG_i = 0
    NDCG_i = 0
    ranked_score = []
    for n in range(len(rankedlist)):
        if rankedlist[n] in testlist:
            HITS_i += 1
            sum_precs += HITS_i / (n + 1.0)
            IDCG_i += 1.0 / math.log2((HITS_i) + 1)
            DCG_i += 1.0 / math.log2((rankedlist.index(rankedlist[n]) + 1) + 1)
            if MRR_i == 0:
                MRR_i = 1.0 / (rankedlist.index(rankedlist[n]) + 1)

        else:
            ranked_score.append(0)
    if HITS_i > 0:
        AP_i = sum_precs / len(testlist)
        HR_i = 1
        NDCG_i = DCG_i / IDCG_i
    len_R = len(rankedlist)
    len_T = len(testlist)
    return AP_i, len_R, len_T, MRR_i, HITS_i, HR_i, NDCG_i


AP = 0
PRE = 0
REC = 0
MRR = 0
HITS = 0
HR = 0
sum_R = 0
sum_T = 0
F1 = 0
NDCG = 0
N = len(R)  # 推荐次数
K = len(R[0])  # 每次推荐的项目个数
for i in range(N):
    AP_i, len_R, len_T, MRR_i, HITS_i, HR_i, NDCG_i = cal_indicators(R[i], T[i])
    AP += AP_i
    sum_R += len_R
    sum_T += len_T
    MRR += MRR_i
    HITS += HITS_i
    HR += HR_i
    NDCG += NDCG_i
PRE = HITS / (sum_R * 1.0)
REC = HITS / (sum_T * 1.0)
F1 = 2 * PRE * REC / (PRE + REC)
HR = HR / (N * 1.0)

MRR = MRR / (N * 1.0)
MAP = AP / (N * 1.0)
NDCG = NDCG / (N * 1.0)
print('评价指标如下:')
print('PRE@', K, ':', PRE)
print('REC@', K, ':', REC)
print('F1@', K, ':', F1)
print('HR@', K, ':', HR)

print('MRR@', K, ':', MRR)
print('MAP@', K, ':', MAP)
print('NDCG@', K, ':', NDCG)

