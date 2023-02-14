##packages
import warnings

warnings.filterwarnings("ignore")
import os
from keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import configparser
from keras.layers import *
import joblib
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection  import GridSearchCV
from xgboost import XGBClassifier
import numpy as np
import shap


# Visualization
import matplotlib.pyplot as plt
import seaborn as sn

#evaluation
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, PrecisionRecallDisplay

def laod_shap_models(results_path, attack_method=None):
    if 'boundary' in attack_method:
        rf_shap_target_rf = joblib.load(results_path + "/rf_shap_rf_np.pkl")
        rf_shap_target_gb = joblib.load(results_path + "/rf_shap_gb_np.pkl")
        xgb_shap_target_rf = joblib.load(results_path + "/xgb_shap_rf_np.pkl")
        xgb_shap_target_gb = joblib.load(results_path + "/xgb_shap_gb_np.pkl")
        mlp_shap_target_rf = joblib.load(results_path + "/mlp_shap_rf_np.pkl")
        mlp_shap_target_gb = joblib.load(results_path + "/mlp_shap_gb_np.pkl")
        return rf_shap_target_rf, rf_shap_target_gb,  xgb_shap_target_rf ,xgb_shap_target_gb \
            ,mlp_shap_target_rf, mlp_shap_target_gb
    else:   
        RF_shap_on_gb_s_t = joblib.load(results_path +"/RF_shap_"+attack_method+"_on_gb_s_t.pkl")
        RF_shap_on_rf_s_t = joblib.load(results_path +"/RF_shap_"+attack_method+"_on_rf_s_t.pkl")
        XGB_shap_on_gb_s_t = joblib.load(results_path +"/XGB_shap_"+attack_method+"_on_gb_s_t.pkl")
        XGB_shap_on_rf_s_t = joblib.load(results_path +"/XGB_shap_"+attack_method+"_on_rf_s_t.pkl")
        MLP_shap_on_gb_s_t = joblib.load(results_path +"/MLP_shap_"+attack_method+"_on_gb_s_t.pkl")
        MLP_shap_on_rf_s_t = joblib.load(results_path +"/MLP_shap_"+attack_method+"_on_rf_s_t.pkl")
        return RF_shap_on_gb_s_t, RF_shap_on_rf_s_t, XGB_shap_on_gb_s_t, XGB_shap_on_rf_s_t, \
            MLP_shap_on_gb_s_t, MLP_shap_on_rf_s_t
    
    

def laod_norm_shap_models(results_path, attack_method=None):
    if 'boundary' in attack_method:
        rf_shap_target_rf = joblib.load(results_path + "/rf_shap_rf_n_np.pkl")
        rf_shap_target_gb = joblib.load(results_path + "/rf_shap_gb_n_np.pkl")
        xgb_shap_target_rf = joblib.load(results_path + "/xgb_shap_rf_n_np.pkl")
        xgb_shap_target_gb = joblib.load(results_path + "/xgb_shap_gb_n_np.pkl")
        mlp_shap_target_rf = joblib.load(results_path + "/mlp_shap_rf_n_np.pkl")
        mlp_shap_target_gb = joblib.load(results_path + "/mlp_shap_gb_n_np.pkl")
        return rf_shap_target_rf, rf_shap_target_gb,  xgb_shap_target_rf ,xgb_shap_target_gb ,mlp_shap_target_rf, mlp_shap_target_gb
    else:
        RF_shap_on_gb_s_t = joblib.load(results_path +"/RF_shap_"+attack_method+"_on_gb_s_t_n.pkl")
        RF_shap_on_rf_s_t = joblib.load(results_path +"/RF_shap_"+attack_method+"_on_rf_s_t_n.pkl")
        XGB_shap_on_gb_s_t = joblib.load(results_path +"/XGB_shap_"+attack_method+"_on_gb_s_t_n.pkl")
        XGB_shap_on_rf_s_t = joblib.load(results_path +"/XGB_shap_"+attack_method+"_on_rf_s_t_n.pkl")
        MLP_shap_on_gb_s_t = joblib.load(results_path +"/MLP_shap_"+attack_method+"_on_gb_s_t_n.pkl")
        MLP_shap_on_rf_s_t = joblib.load(results_path +"/MLP_shap_"+attack_method+"_on_rf_s_t_n.pkl")
        return RF_shap_on_gb_s_t, RF_shap_on_rf_s_t, XGB_shap_on_gb_s_t, XGB_shap_on_rf_s_t, \
            MLP_shap_on_gb_s_t, MLP_shap_on_rf_s_t

    


