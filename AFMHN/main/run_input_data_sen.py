import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.models import basicFM, DeepFM, AFM, NAFM, xDeepFM, AutoInt, xDeepAFM, xDeepMAFM, DCNMix, AFN
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

# 设置Seed值
setup_seed(2022)


if __name__ == "__main__":

    # load input data
    # with open('input_data.pickle', 'rb') as f:
    #     input_data = pickle.load(f)
    # dnn_feature_columns=input_data["dnn_feature_columns"]
    # linear_feature_columns=input_data["linear_feature_columns"]
    # train=input_data["train"]
    # test=input_data["test"]
    # train_model_input=input_data["train_model_input"]
    # test_model_input=input_data["test_model_input"]
    # target = ['label']

    data = pd.read_csv('../result/input_data_sen_all.csv')
    sparse_features = ['C1', 'C3']
    # sparse_features = ['C' + str(i) for i in range(1, 5)]
    dense_features = ['I' + str(i) for i in range(1, 104)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=2)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    # default velues of parameters
    # [dnn_hidden_units, attention_factor, cin_layer_size, l2_reg, dropout, batch_size] = [8, 16, 8, 0.1, 0.1, 256]
    [dnn_hidden_units, attention_factor, cin_layer_size, l2_reg, dropout, batch_size] = [32, 8, 16, 0.1, 0.1, 128]
    # [dnn_hidden_units, attention_factor, cin_layer_size, l2_reg, dropout, batch_size] = [2, 4, 4, 0.1, 0.5, 64]
    df = pd.DataFrame(np.zeros((10, 8)), index=["basicFM", "DeepFM", "AutoInt", "xDeepFM", "AFM", "NAFM", "xDeepAFM", "xDeepMAFM","DCNMix","AFN"],
                      columns=["LogLoss_20%", "LogLoss_40%", "LogLoss_60%", "LogLoss_80%", "AUC_20%", "AUC_40%",
                               "AUC_60%", "AUC_80%", ])

    for test_size in [0.2, 0.4, 0.6, 0.8]:
        proportion = int(test_size * 100)
        train, test = train_test_split(data, test_size=test_size)
        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        # 4.Define Model,train,predict and evaluate

        # Evaluate basicFM
        # LOSS = [];
        # AUC = []
        # for i in range(5):
        #     model = basicFM(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg,
        #                     task='binary')
        #     model.compile("adam", "binary_crossentropy",
        #                   metrics=['binary_crossentropy'], )
        #     history = model.fit(train_model_input, train[target].values,
        #                         batch_size=batch_size, epochs=10, verbose=2, validation_split=0.2, )
        #     pred_ans = model.predict(test_model_input, batch_size=256)
        #     loss_value = log_loss(test[target].values, pred_ans)
        #     auc_value = roc_auc_score(test[target].values, pred_ans)
        #     LOSS.append(loss_value);
        #     AUC.append(auc_value)
        # df.loc["basicFM", "LogLoss_{}%".format(proportion)] = round(np.mean(LOSS), 4)
        # df.loc["basicFM", "AUC_{}%".format(proportion)] = round(np.mean(AUC), 4)

        # Evaluate DeepFM
        # LOSS = [];
        # AUC = []
        # for i in range(5):
        #     model = DeepFM(linear_feature_columns, dnn_feature_columns,
        #                    dnn_hidden_units=(dnn_hidden_units, dnn_hidden_units),
        #                    l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, l2_reg_dnn=l2_reg, dnn_dropout=dropout,
        #                    task='binary')
        #     model.compile("adam", "binary_crossentropy",
        #                   metrics=['binary_crossentropy'], )
        #     history = model.fit(train_model_input, train[target].values,
        #                         batch_size=batch_size, epochs=10, verbose=2, validation_split=0.2, )
        #     pred_ans = model.predict(test_model_input, batch_size=256)
        #     loss_value = log_loss(test[target].values, pred_ans)
        #     auc_value = roc_auc_score(test[target].values, pred_ans)
        #     LOSS.append(loss_value);
        #     AUC.append(auc_value)
        # df.loc["DeepFM", "LogLoss_{}%".format(proportion)] = round(np.mean(LOSS), 4)
        # df.loc["DeepFM", "AUC_{}%".format(proportion)] = round(np.mean(AUC), 4)

        # Evaluate AFM
        # LOSS = [];
        # AUC = []
        # for i in range(5):
        #     model = AFM(linear_feature_columns, dnn_feature_columns, attention_factor=attention_factor,
        #                 l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, l2_reg_att=l2_reg, afm_dropout=dropout,
        #                 task='binary')
        #     model.compile("adam", "binary_crossentropy",
        #                   metrics=['binary_crossentropy'], )
        #     history = model.fit(train_model_input, train[target].values,
        #                         batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
        #     pred_ans = model.predict(test_model_input, batch_size=256)
        #     loss_value = log_loss(test[target].values, pred_ans)
        #     auc_value = roc_auc_score(test[target].values, pred_ans)
        #     LOSS.append(loss_value);
        #     AUC.append(auc_value)
        # df.loc["AFM", "LogLoss_{}%".format(proportion)] = round(np.mean(LOSS), 4)
        # df.loc["AFM", "AUC_{}%".format(proportion)] = round(np.mean(AUC), 4)

        # Evaluate NAFM
        # LOSS = [];
        # AUC = []
        # for i in range(5):
        #     model = NAFM(linear_feature_columns, dnn_feature_columns, attention_factor=attention_factor,
        #                  dnn_hidden_units=(dnn_hidden_units, dnn_hidden_units), l2_reg_linear=l2_reg,
        #                  l2_reg_embedding=l2_reg,
        #                  l2_reg_dnn=l2_reg, l2_reg_att=l2_reg, dnn_dropout=dropout, afm_dropout=dropout,
        #                  bi_dropout=dropout, task='binary')
        #     model.compile("adam", "binary_crossentropy",
        #                   metrics=['binary_crossentropy'], )
        #     history = model.fit(train_model_input, train[target].values,
        #                         batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
        #     pred_ans = model.predict(test_model_input, batch_size=256)
        #     loss_value = log_loss(test[target].values, pred_ans)
        #     auc_value = roc_auc_score(test[target].values, pred_ans)
        #     LOSS.append(loss_value);
        #     AUC.append(auc_value)
        # df.loc["NAFM", "LogLoss_{}%".format(proportion)] = round(np.mean(LOSS), 4)
        # df.loc["NAFM", "AUC_{}%".format(proportion)] = round(np.mean(AUC), 4)

        # Evaluate xDeepFM
        # LOSS = [];
        # AUC = []
        # for i in range(5):
        #     model = xDeepFM(linear_feature_columns, dnn_feature_columns,
        #                     dnn_hidden_units=(dnn_hidden_units, dnn_hidden_units),
        #                     cin_layer_size=(cin_layer_size, cin_layer_size),
        #                     l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, l2_reg_dnn=l2_reg, dnn_dropout=dropout,
        #                     task='binary')
        #     model.compile("adam", "binary_crossentropy",
        #                   metrics=['binary_crossentropy'], )
        #     history = model.fit(train_model_input, train[target].values,
        #                         batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
        #     pred_ans = model.predict(test_model_input, batch_size=256)
        #     loss_value = log_loss(test[target].values, pred_ans)
        #     auc_value = roc_auc_score(test[target].values, pred_ans)
        #     LOSS.append(loss_value);
        #     AUC.append(auc_value)
        # df.loc["xDeepFM", "LogLoss_{}%".format(proportion)] = round(np.mean(LOSS), 4)
        # df.loc["xDeepFM", "AUC_{}%".format(proportion)] = round(np.mean(AUC), 4)

        # Evaluate xDeepAFM
        LOSS = [];
        AUC = []
        for i in range(5):
            model = xDeepAFM(linear_feature_columns, dnn_feature_columns,
                             dnn_hidden_units=(dnn_hidden_units, dnn_hidden_units),
                             cin_layer_size=(cin_layer_size, cin_layer_size), attention_factor=attention_factor,
                             l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, l2_reg_dnn=l2_reg,
                             l2_reg_cin=l2_reg, l2_reg_att=l2_reg,
                             dnn_dropout=dropout, bi_dropout=dropout, afm_dropout=dropout,
                             task='binary')
            model.compile("adam", "binary_crossentropy",
                          metrics=['binary_crossentropy'], )
            history = model.fit(train_model_input, train[target].values,
                                batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
            pred_ans = model.predict(test_model_input, batch_size=256)
            loss_value = log_loss(test[target].values, pred_ans)
            auc_value = roc_auc_score(test[target].values, pred_ans)
            LOSS.append(loss_value);
            AUC.append(auc_value)
        df.loc["xDeepAFM", "LogLoss_{}%".format(proportion)] = round(np.mean(LOSS), 4)
        df.loc["xDeepAFM", "AUC_{}%".format(proportion)] = round(np.mean(AUC), 4)

        # Evaluate xDeepMAFM
        # LOSS = [];
        # AUC = []
        # for i in range(5):
        #     model = xDeepMAFM(linear_feature_columns, dnn_feature_columns,
        #                      dnn_hidden_units=(dnn_hidden_units, dnn_hidden_units),
        #                      cin_layer_size=(cin_layer_size, cin_layer_size),att_head_num=2, att_layer_num=3,
        #                      l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, l2_reg_dnn=l2_reg,
        #                      l2_reg_cin=l2_reg,
        #                      dnn_dropout=dropout, bi_dropout=dropout,
        #                      task='binary')
        #     model.compile("adam", "binary_crossentropy",
        #                   metrics=['binary_crossentropy'], )
        #     history = model.fit(train_model_input, train[target].values,
        #                         batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
        #     pred_ans = model.predict(test_model_input, batch_size=256)
        #     loss_value = log_loss(test[target].values, pred_ans)
        #     auc_value = roc_auc_score(test[target].values, pred_ans)
        #     LOSS.append(loss_value);
        #     AUC.append(auc_value)
        # df.loc["xDeepMAFM", "LogLoss_{}%".format(proportion)] = round(np.mean(LOSS), 4)
        # df.loc["xDeepMAFM", "AUC_{}%".format(proportion)] = round(np.mean(AUC), 4)

        # Evaluate AutoInt
        # LOSS = [];
        # AUC = []
        # for i in range(5):
        #     model = AutoInt(linear_feature_columns, dnn_feature_columns,
        #                     dnn_hidden_units=(dnn_hidden_units, dnn_hidden_units),
        #                     l2_reg_embedding=l2_reg, l2_reg_dnn=l2_reg,
        #                     dnn_dropout=dropout,
        #                     task='binary')
        #     model.compile("adam", "binary_crossentropy",
        #                   metrics=['binary_crossentropy'], )
        #     history = model.fit(train_model_input, train[target].values,
        #                         batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
        #     pred_ans = model.predict(test_model_input, batch_size=256)
        #     loss_value = log_loss(test[target].values, pred_ans)
        #     auc_value = roc_auc_score(test[target].values, pred_ans)
        #     LOSS.append(loss_value);
        #     AUC.append(auc_value)
        # df.loc["AutoInt", "LogLoss_{}%".format(proportion)] = round(np.mean(LOSS), 4)
        # df.loc["AutoInt", "AUC_{}%".format(proportion)] = round(np.mean(AUC), 4)

        # # Evaluate DCNMix
        # LOSS = [];
        # AUC = []
        # for i in range(5):
        #     model = DCNMix(linear_feature_columns, dnn_feature_columns,
        #                  dnn_hidden_units=(128, 128),
        #                  l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, l2_reg_dnn=l2_reg,l2_reg_cross=l2_reg,
        #                  dnn_dropout=0.1,
        #                  task='binary')
        #     model.compile("adam", "binary_crossentropy",
        #                   metrics=['binary_crossentropy'], )
        #     history = model.fit(train_model_input, train[target].values,
        #                         batch_size=64, epochs=100, verbose=2, validation_split=0.2, )
        #     pred_ans = model.predict(test_model_input, batch_size=256)
        #     loss_value = log_loss(test[target].values, pred_ans)
        #     auc_value = roc_auc_score(test[target].values, pred_ans)
        #     LOSS.append(loss_value);
        #     AUC.append(auc_value)
        # df.loc["DCNMix", "LogLoss_{}%".format(proportion)] = round(np.mean(LOSS), 4)
        # df.loc["DCNMix", "AUC_{}%".format(proportion)] = round(np.mean(AUC), 4)

        # Evaluate AFN
        # LOSS = [];
        # AUC = []
        # for i in range(5):
        #     model = AFN(linear_feature_columns, dnn_feature_columns,
        #                 ltl_hidden_size=32,
        #                  afn_dnn_hidden_units=(8, 8),
        #                  l2_reg_linear=0.001, l2_reg_embedding=0.001, l2_reg_dnn=0.001,
        #                  dnn_dropout=0.5,
        #                  task='binary')
        #     model.compile("adam", "binary_crossentropy",
        #                   metrics=['binary_crossentropy'], )
        #     history = model.fit(train_model_input, train[target].values,
        #                         batch_size=64, epochs=100, verbose=2, validation_split=0.2, )
        #     pred_ans = model.predict(test_model_input, batch_size=256)
        #     loss_value = log_loss(test[target].values, pred_ans)
        #     auc_value = roc_auc_score(test[target].values, pred_ans)
        #     LOSS.append(loss_value);
        #     AUC.append(auc_value)
        # df.loc["AFN", "LogLoss_{}%".format(proportion)] = round(np.mean(LOSS), 4)
        # df.loc["AFN", "AUC_{}%".format(proportion)] = round(np.mean(AUC), 4)

    df.to_csv("../result/evaluation_results_sen_1129.csv", index=True, float_format='%.4f')
    print(df)
