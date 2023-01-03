import numpy as np
import pickle
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.models import xDeepAFM
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
    with open('../result/input_data_new.pickle', 'rb') as f:
        input_data = pickle.load(f)
    dnn_feature_columns = input_data["dnn_feature_columns"]
    linear_feature_columns = input_data["linear_feature_columns"]
    train = input_data["train"]
    test = input_data["test"]
    train_model_input = input_data["train_model_input"]
    test_model_input = input_data["test_model_input"]
    target = ['label']

    p_dnn_hidden_units = [2, 4, 8, 16, 32, 64, 128, 256]
    p_attention_factor = [2, 4, 8, 16, 32, 64, 128, 256]
    p_cin_layer_size = [2, 4, 8, 16, 32, 64, 128, 256]
    p_l2_reg = [1e-1, 1e-2, 1e-3]
    p_dropout = [0.1, 0.5]
    P_batch_size = [64, 128, 256]
    all_paramaters = []
    for dnn_hidden_units in p_dnn_hidden_units:
        for attention_factor in p_attention_factor:
            for cin_layer_size in p_cin_layer_size:
                for l2_reg in p_l2_reg:
                    for dropout in p_dropout:
                        for batch_size in P_batch_size:
                            all_paramaters.append(
                                (dnn_hidden_units, attention_factor, cin_layer_size, l2_reg, dropout, batch_size))
    best_loss = np.inf;
    best_auc = 0
    for index, (dnn_hidden_units, attention_factor, cin_layer_size, l2_reg, dropout, batch_size) in enumerate(
            all_paramaters):
        model = xDeepAFM(linear_feature_columns, dnn_feature_columns, attention_factor=attention_factor,
                         dnn_hidden_units=(dnn_hidden_units, dnn_hidden_units),
                         cin_layer_size=(cin_layer_size, cin_layer_size),
                         l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, l2_reg_att=l2_reg, l2_reg_cin=l2_reg,l2_reg_dnn=l2_reg,
                         dnn_dropout=dropout, afm_dropout=dropout, bi_dropout=dropout,
                         task='binary')
        model.compile("adam", "binary_crossentropy",
                      metrics=['binary_crossentropy'], )
        history = model.fit(train_model_input, train[target].values,
                            batch_size=batch_size, epochs=100, verbose=2, validation_split=0.2, )
        pred_ans = model.predict(test_model_input, batch_size=256)
        loss_value = round(log_loss(test[target].values, pred_ans), 4)
        auc_value = round(roc_auc_score(test[target].values, pred_ans), 4)
        if loss_value <= best_loss and auc_value >= best_auc:
            best_loss = loss_value;
            best_auc = auc_value
            best_parameters = [dnn_hidden_units, attention_factor, cin_layer_size, l2_reg, dropout, batch_size]
        print("*****:", index, "-->", len(all_paramaters), best_loss, best_auc, best_parameters)
    print()
    print(best_loss, best_auc, best_parameters)
