import numpy as np
import pandas as pd

#from Utils.anomaly_utils import get_ocsvm
from Shap_Utils import pre_proccess_utils as pre
from Shap_Utils import eval_utils as eval
from Shap_Utils import load_utils as load
from Shap_Utils import train_utils as train

x_train_target, y_train_target, x_test, y_test,


gb_shap_train, rf_shap_train, gb_shap_test, rf_shap_test = \
                    pre.get_shap_dataset(pred_col_name, dataset_file, df_results_path)
            
gb_shap_train, rf_shap_train, gb_shap_test, rf_shap_test = \
                    pre.normalaize_shap_dataset(pred_col_name, dataset_file, df_results_path)
            
train.train_models(pred_col_name, gb_shap_train, rf_shap_train, df_results_path, dataset_file)


rf_shap_rf, rf_shap_gb,  xgb_shap_rf ,xgb_shap_gb ,mlp_shap_rf, mlp_shap_gb = \
        load.laod_shap_models(df_results_path)
#rf_shap_rf, rf_shap_gb,  xgb_shap_rf ,xgb_shap_gb ,mlp_shap_rf, mlp_shap_gb = \
#        load.laod_norm_shap_models(df_results_path) 

x_gb_shap_test, y_gb_shap_test = pre.split_x_y(gb_shap_test, pred_col_name)
x_rf_shap_test, y_rf_shap_test = pre.split_x_y(rf_shap_test, pred_col_name)
x_gb_shap_train, y_gb_shap_train = pre.split_x_y(gb_shap_train, pred_col_name)
x_rf_shap_train, y_rf_shap_train = pre.split_x_y(rf_shap_train, pred_col_name)

f1, acc = eval.eval_models(rf_shap_gb, x_rf_shap_train, y_rf_shap_train, x_rf_shap_test, y_rf_shap_test, 'rf')
f1, acc = eval.eval_models(rf_shap_rf, x_rf_shap_train, y_rf_shap_train, x_rf_shap_test, y_rf_shap_test, 'rf')
f1, acc = eval.eval_models(rf_shap_gb, x_gb_shap_train, y_gb_shap_train, x_gb_shap_test, y_gb_shap_test, 'rf')

f1, acc = eval.eval_models(xgb_shap_rf, x_rf_shap_train, y_rf_shap_train, x_rf_shap_test, y_rf_shap_test, 'xgb')
f1, acc = eval.eval_models(xgb_shap_gb, x_gb_shap_train, y_gb_shap_train, x_gb_shap_test, y_gb_shap_test, 'xgb')
f1, acc = eval.eval_models(mlp_shap_rf, x_rf_shap_train, y_rf_shap_train, x_rf_shap_test, y_rf_shap_test, 'nn')
f1, acc = eval.eval_models(mlp_shap_gb, x_gb_shap_train, y_gb_shap_train, x_gb_shap_test, y_gb_shap_test, 'nn')