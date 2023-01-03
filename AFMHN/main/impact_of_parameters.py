import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.models import basicFM, DeepFM, NFM, AFM, xDeepAFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

if __name__ == "__main__":
    # load input data
    with open('../result/input_data_sen_all_1128.pickle', 'rb') as f:
        input_data = pickle.load(f)
    dnn_feature_columns = input_data["dnn_feature_columns"]
    linear_feature_columns = input_data["linear_feature_columns"]
    train = input_data["train"]
    test = input_data["test"]
    train_model_input = input_data["train_model_input"]
    test_model_input = input_data["test_model_input"]
    target = ['label']

    # default velues of parameters
    # [dnn_hidden_units,attention_factor,l2_reg,dropout,batch_size]=[64, 32, 0.001, 0.1, 256]
    [dnn_hidden_units, attention_factor, cin_layer_size, l2_reg, dropout, batch_size] = [32, 8, 16, 0.1, 0.1, 256]

    p_dnn_hidden_units = [2, 4, 8, 16, 32, 64, 128]
    p_attention_factor = [2, 4, 8, 16, 32, 64, 128]
    p_cin_layer_size = [2, 4, 8, 16, 32, 64, 128]
    p_l2_reg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]
    p_dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    df_loss = pd.DataFrame(np.zeros((5, 9)),
                           index=["dnn_hidden_units", "cin_layer_size", "attention_factor", "l2_reg", "dropout"])
    df_auc = pd.DataFrame(np.zeros((5, 9)),
                          index=["dnn_hidden_units", "cin_layer_size", "attention_factor", "l2_reg", "dropout"])

    # inpact of dnn_hidden_units
    # for index, parameter in enumerate(p_dnn_hidden_units):
    #     dnn_hidden_units = parameter
    #     LOSS = [];
    #     AUC = []
    #     for i in range(5):
    #         model = xDeepAFM(linear_feature_columns, dnn_feature_columns,
    #                          dnn_hidden_units=(dnn_hidden_units, dnn_hidden_units),
    #                          cin_layer_size=(cin_layer_size, cin_layer_size), attention_factor=attention_factor,
    #                          l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, l2_reg_dnn=l2_reg,
    #                          l2_reg_cin=l2_reg, l2_reg_att=l2_reg,
    #                          dnn_dropout=dropout, bi_dropout=dropout, afm_dropout=dropout,
    #                          task='binary')
    #         model.compile("adam", "binary_crossentropy",
    #                       metrics=['binary_crossentropy'], )
    #         history = model.fit(train_model_input, train[target].values,
    #                             batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
    #         pred_ans = model.predict(test_model_input, batch_size=256)
    #         loss_value = log_loss(test[target].values, pred_ans)
    #         auc_value = roc_auc_score(test[target].values, pred_ans)
    #         LOSS.append(loss_value);
    #         AUC.append(auc_value)
    #     df_loss.loc["dnn_hidden_units", index] = round(np.mean(LOSS), 4)
    #     df_auc.loc["dnn_hidden_units", index] = round(np.mean(AUC), 4)

    # inpact of cin_layer_size
    # for index, parameter in enumerate(p_cin_layer_size):
    #     cin_layer_size = parameter
    #     LOSS = [];
    #     AUC = []
    #     for i in range(5):
    #         model = xDeepAFM(linear_feature_columns, dnn_feature_columns,
    #                          dnn_hidden_units=(dnn_hidden_units, dnn_hidden_units),
    #                          cin_layer_size=(cin_layer_size, cin_layer_size), attention_factor=attention_factor,
    #                          l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, l2_reg_dnn=l2_reg,
    #                          l2_reg_cin=l2_reg, l2_reg_att=l2_reg,
    #                          dnn_dropout=dropout, bi_dropout=dropout, afm_dropout=dropout,
    #                          task='binary')
    #         model.compile("adam", "binary_crossentropy",
    #                       metrics=['binary_crossentropy'], )
    #         history = model.fit(train_model_input, train[target].values,
    #                             batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
    #         pred_ans = model.predict(test_model_input, batch_size=256)
    #         loss_value = log_loss(test[target].values, pred_ans)
    #         auc_value = roc_auc_score(test[target].values, pred_ans)
    #         LOSS.append(loss_value);
    #         AUC.append(auc_value)
    #     df_loss.loc["cin_layer_size", index] = round(np.mean(LOSS), 4)
    #     df_auc.loc["cin_layer_size", index] = round(np.mean(AUC), 4)

    # inpact of attention_factor
    # for index, parameter in enumerate(p_attention_factor):
    #     attention_factor = parameter
    #     LOSS = [];
    #     AUC = []
    #     for i in range(5):
    #         model = xDeepAFM(linear_feature_columns, dnn_feature_columns,
    #                          dnn_hidden_units=(dnn_hidden_units, dnn_hidden_units),
    #                          cin_layer_size=(cin_layer_size, cin_layer_size), attention_factor=attention_factor,
    #                          l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, l2_reg_dnn=l2_reg,
    #                          l2_reg_cin=l2_reg, l2_reg_att=l2_reg,
    #                          dnn_dropout=dropout, bi_dropout=dropout, afm_dropout=dropout,
    #                          task='binary')
    #         model.compile("adam", "binary_crossentropy",
    #                       metrics=['binary_crossentropy'], )
    #         history = model.fit(train_model_input, train[target].values,
    #                             batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
    #         pred_ans = model.predict(test_model_input, batch_size=256)
    #         loss_value = log_loss(test[target].values, pred_ans)
    #         auc_value = roc_auc_score(test[target].values, pred_ans)
    #         LOSS.append(loss_value);
    #         AUC.append(auc_value)
    #     df_loss.loc["attention_factor", index] = round(np.mean(LOSS), 4)
    #     df_auc.loc["attention_factor", index] = round(np.mean(AUC), 4)

    # inpact of l2_reg
    # for index, parameter in enumerate(p_l2_reg):
    #     l2_reg = parameter
    #     LOSS = [];
    #     AUC = []
    #     for i in range(5):
    #         model = xDeepAFM(linear_feature_columns, dnn_feature_columns,
    #                          dnn_hidden_units=(dnn_hidden_units, dnn_hidden_units),
    #                          cin_layer_size=(cin_layer_size, cin_layer_size), attention_factor=attention_factor,
    #                          l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, l2_reg_dnn=l2_reg,
    #                          l2_reg_cin=l2_reg, l2_reg_att=l2_reg,
    #                          dnn_dropout=dropout, bi_dropout=dropout, afm_dropout=dropout,
    #                          task='binary')
    #         model.compile("adam", "binary_crossentropy",
    #                       metrics=['binary_crossentropy'], )
    #         history = model.fit(train_model_input, train[target].values,
    #                             batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
    #         pred_ans = model.predict(test_model_input, batch_size=256)
    #         loss_value = log_loss(test[target].values, pred_ans)
    #         auc_value = roc_auc_score(test[target].values, pred_ans)
    #         LOSS.append(loss_value);
    #         AUC.append(auc_value)
    #     df_loss.loc["l2_reg", index] = round(np.mean(LOSS), 4)
    #     df_auc.loc["l2_reg", index] = round(np.mean(AUC), 4)

    # inpact of dropout
    for index, parameter in enumerate(p_dropout):
        dropout = parameter
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
        df_loss.loc["dropout", index] = round(np.mean(LOSS), 4)
        df_auc.loc["dropout", index] = round(np.mean(AUC), 4)

    # save resutls to files
    df_loss.to_csv("../result/Logloss_impact_of_parameters_1129_1.csv", index=True, float_format='%.4f')
    df_auc.to_csv("../result/AUC_impact_of_parameters_1129_1.csv", index=True, float_format='%.4f')
