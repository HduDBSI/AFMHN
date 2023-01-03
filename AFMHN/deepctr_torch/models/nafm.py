# -*- coding:utf-8 -*-
"""
To use NAFM, you have to do the following steps:
1. download the deepctr package from https://github.com/shenweichen/DeepCTR first
2. put this file to the directory: DeepCTR\deepctr\models
3. add NAFM model to the file: DeepCTR\deepctr\models\__init__.py
4. after doing the above 3 steps, you can use any complex model with model.fit()and model.predict()
you can refer to some examples on how to use models in DeepCTR package from the website: https://deepctr-doc.readthedocs.io/en/latest/Examples.html#
"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN, BiInteractionPooling
from ..layers import FM, AFMLayer


class NAFM(BaseModel):
    """Instantiates the Attentional Factorization Machine architecture.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param use_attention: bool,whether use attention or not,if set to ``False``.it is the same as **standard Factorization Machine**
    :param attention_factor: positive integer,units in attention net
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float . L2 regularizer strength applied to DNN
    :param l2_reg_att: float. L2 regularizer strength applied to attention net
    :param biout_dropout: When not ``None``, the probability we will drop out the output of BiInteractionPooling Layer.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param afm_dropout: float in [0,1), Fraction of the attention net output units to dropout.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_activation: Activation function to use in deep net
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128), use_attention=True,
                 attention_factor=8,
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, l2_reg_att=1e-5, afm_dropout=0,
                 init_std=0.0001, seed=1024, bi_dropout=0,
                 dnn_dropout=0, dnn_activation='relu', task='binary', device='cpu', gpus=None):
        super(NAFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                   l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                   device=device, gpus=gpus)

        self.use_attention = use_attention
        if use_attention:
            self.fm = AFMLayer(self.embedding_size, attention_factor, l2_reg_att, afm_dropout,
                               seed, device)
            self.add_regularization_weight(self.fm.attention_W, l2=l2_reg_att)
        else:
            self.fm = FM()

        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, include_sparse=False) + self.embedding_size,
                       dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False,
                       init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.bi_pooling = BiInteractionPooling()
        self.bi_dropout = bi_dropout
        if self.bi_dropout > 0:
            self.dropout = nn.Dropout(bi_dropout)
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict, support_dense=True)
        linear_logit = self.linear_model(X)

        fm_input = torch.cat(sparse_embedding_list, dim=1)
        bi_out = self.bi_pooling(fm_input)
        if self.bi_dropout:
            bi_out = self.dropout(bi_out)

        if self.use_dnn:
            dnn_input = combined_dnn_input([bi_out], dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)

        if self.use_attention:
            fm_logit = self.fm(sparse_embedding_list)
        else:
            fm_logit = self.fm(torch.cat(sparse_embedding_list, dim=1))

        logit = linear_logit + dnn_logit + fm_logit

        y_pred = self.out(logit)

        return y_pred
