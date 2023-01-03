#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-12-02 09:29:18
# @Author  : Guosheng Kang (guoshengkang@gmail.com)
# @Link    : https://guoshengkang.github.io
# @Version : $Id$

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

p_dnn_hidden_units = [2, 4, 8, 16, 32, 64, 128]
p_attention_factor = [2, 4, 8, 16, 32, 64, 128]
p_cin_layer_size = [2, 4, 8, 16, 32, 64, 128]
p_l2_reg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]
p_dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

df_loss = pd.read_csv('../result/Logloss_impact_of_parameters_1129.csv', index_col=0)
df_auc = pd.read_csv('../result/AUC_impact_of_parameters_1129.csv', index_col=0)

fig=plt.figure(figsize=(13, 6))

# Hidden Units
ax1=fig.add_subplot(2,5,1)
ax1.plot(range(7),df_loss.loc["dnn_hidden_units",],"bv-")
plt.xlabel("DNN Hidden Units\n(a)")
plt.ylabel("Logloss")
plt.xlim((0,6))
plt.xticks(range(7),["2","4","8","16","32","64","128"])
plt.grid(linestyle='-.')

ax2=fig.add_subplot(2,5,6)
ax2.plot(range(7),df_auc.loc["dnn_hidden_units",],"r^-")
plt.xlabel("DNN Hidden Units\n(f)")
plt.ylabel("AUC")
plt.xlim((0,6))
plt.xticks(range(7),["2","4","8","16","32","64","128"])
plt.grid(linestyle='-.')

# Cin_layer_size
ax1=fig.add_subplot(2,5,2)
ax1.plot(range(7),df_loss.loc["cin_layer_size",],"bv-")
plt.xlabel("CIN Layer Size\n(b)")
plt.ylabel("Logloss")
plt.xlim((0,6))
plt.xticks(range(7),["2","4","8","16","32","64","128"])
plt.grid(linestyle='-.')

ax2=fig.add_subplot(2,5,7)
ax2.plot(range(7),df_auc.loc["cin_layer_size",],"r^-")
plt.xlabel("CIN Layer Size\n(g)")
plt.ylabel("AUC")
plt.xlim((0,6))
plt.xticks(range(7),["2","4","8","16","32","64","128"])
plt.grid(linestyle='-.')

# Attention Factors
ax1=fig.add_subplot(2,5,3)
ax1.plot(range(7),df_loss.loc["attention_factor",],"bv-")
plt.xlabel("Attention Factor\n(c)")
plt.ylabel("Logloss")
plt.xlim((0,6))
plt.xticks(range(7),["2","4","8","16","32","64","128"])
plt.grid(linestyle='-.')

ax2=fig.add_subplot(2,5,8)
ax2.plot(range(7),df_auc.loc["attention_factor",],"r^-")
plt.xlabel("Attention Factor\n(h)")
plt.ylabel("AUC")
plt.xlim((0,6))
plt.xticks(range(7),["2","4","8","16","32","64","128"])
plt.grid(linestyle='-.')

# L2 Regularization
ax1=fig.add_subplot(2,5,4)
ax1.plot(range(7),df_loss.loc["l2_reg",],"bv-")
plt.xlabel("L2 Regularization\n(d)")
plt.ylabel("Logloss")
plt.xlim((0,6))
plt.xticks(range(7),["1e-6","1e-5","1e-4","1e-3","1e-2","1e-1","0"])
plt.grid(linestyle='-.')

ax2=fig.add_subplot(2,5,9)
ax2.plot(range(7),df_auc.loc["l2_reg",],"r^-")
plt.xlabel("L2 Regularization\n(i)")
plt.ylabel("AUC")
plt.xlim((0,6))
plt.xticks(range(7),["1e-6","1e-5","1e-4","1e-3","1e-2","1e-1","0"])
plt.grid(linestyle='-.')

# Dropout Rate
ax1=fig.add_subplot(2,5,5)
ax1.plot(range(7),df_loss.loc["dropout",],"bv-")
plt.xlabel("Dropout Rate\n(e)")
plt.ylabel("Logloss")
plt.xlim((0,6))
plt.xticks(range(7),["0","0.1","0.2","0.3","0.4","0.5","0.6"])
plt.grid(linestyle='-.')

ax2=fig.add_subplot(2,5,10)
ax2.plot(range(7),df_auc.loc["dropout",],"r^-")
plt.xlabel("Dropout Rate\n(j)")
plt.ylabel("AUC")
plt.xlim((0,6))
plt.xticks(range(7),["0","0.1","0.2","0.3","0.4","0.5","0.6"])
plt.grid(linestyle='-.')

plt.tight_layout() #设置默认的间距
plt.savefig('../result/impact_of_parameters_1129.png', dpi=200) #指定分辨率保存
plt.show()
