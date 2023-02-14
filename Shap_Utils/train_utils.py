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
import config


# Visualization
import matplotlib.pyplot as plt
import seaborn as sn

#evaluation
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, PrecisionRecallDisplay

def train_rf_model(x_train, y_train, dataset_file):

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    if( "HATE" in dataset_file):
        param_test1 = {'n_estimators': range(5, 100, 5), 'max_depth': range(1, 10, 2)}
    else:
        param_test1 = {'n_estimators': range(5, 100, 5), 'max_depth': range(1, 20, 1)}
    gsearch = GridSearchCV(
        estimator=RandomForestClassifier(random_state=1234), #learningrate??
        param_grid=param_test1, scoring='accuracy', n_jobs=-1, iid=False, cv=5)
    print("RF: performing hyperparameter tuning using GridSearchCV, cv=5...")
    gsearch.fit(np.array(x_train), np.array(y_train))
    print("Best params found: " + str(gsearch.best_params_))
    gsearch.best_estimator_.fit(np.array(x_train), np.array(y_train))
    return gsearch.best_estimator_

def train_xgb_model(x_train, y_train):
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    estimator = XGBClassifier(objective= 'binary:hinge', nthread=4, seed=42, use_label_encoder=False)
    parameters = {'max_depth': range (1, 10, 1), 'n_estimators': range(5, 100, 5), 'learning_rate': [0.5, 0.1, 0.01, 0.05]}
    '''
    if( "HATE" in dataset_file):
        param_test1 = {'n_estimators': range(20, 100, 10), 'max_depth': range(1, 5, 1)}#10
    else:
        param_test1 = {'n_estimators': range(20, 1000, 10), 'max_depth': range(1, 5, 1)}#30
    '''
    gsearch = GridSearchCV(
        estimator = estimator, param_grid=parameters, scoring='accuracy',  n_jobs=-1, iid=False, cv=5)
    
    #gsearch = GridSearchCV(
    #    estimator=HistGradientBoostingClassifier(learning_rate=0.1, random_state=1234),
    #    param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    
    print("XGB: performing hyperparameter tuning using GridSearchCV, cv=5...")
    gsearch.fit(x_train, y_train)
    print("Best params found: " + str(gsearch.best_params_))
    gsearch.best_estimator_.fit(x_train, y_train)
    return gsearch.best_estimator_

def train_mlp_model(x_train, y_train):

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    mlp = MLPClassifier(max_iter=1000)
    parameter_space = {
        'hidden_layer_sizes': [(10,10),(10,10,10),(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.00001, 0.0001, 0.001, 0.05],
        'learning_rate': ['constant','adaptive']
        }

    gsearch = GridSearchCV(estimator=mlp, param_grid=parameter_space, n_jobs=-1, cv=5, scoring='accuracy')
    
    print("MLP: performing hyperparameter tuning using GridSearchCV, cv=5...")
    gsearch.fit(x_train, y_train)
    print("Best params found: " + str(gsearch.best_params_))
    gsearch.best_estimator_.fit(x_train, y_train)
    
    return gsearch.best_estimator_

def train_models(pred_col_name, gb_shap_train, rf_shap_train, dataset_path, dataset_file, attack_method, model_type=None):
    
    # preparing x_train, y_train
    y_gb_shap_train = gb_shap_train[pred_col_name]#.values
    y_rf_shap_train = rf_shap_train[pred_col_name]#.values
    
    x_gb_shap_train, x_rf_shap_train = [x.drop([pred_col_name], axis=1) for x in [gb_shap_train, rf_shap_train]]
    
    print('training RF models...')
    rf_shap_target_gb = train_rf_model(x_gb_shap_train, y_gb_shap_train, dataset_file) 
    rf_shap_target_rf = train_rf_model(x_rf_shap_train, y_rf_shap_train, dataset_file) 
            
    print ('saving RF models')
    if (model_type == 'None'):
        joblib.dump(rf_shap_target_rf, dataset_path+"/RF_shap_rf_n_np.pkl")
        joblib.dump(rf_shap_target_gb, dataset_path+"/RF_shap_gb_n_np.pkl")
    elif(model_type == 'rfgb'):    
        joblib.dump(rf_shap_target_rf, dataset_path+"/RF_shap_"+attack_method+"_on_rf_s_t_n.pkl")
        joblib.dump(rf_shap_target_gb, dataset_path+"/RF_shap_"+attack_method+"_on_gb_s_t_n.pkl")
    else:    
        joblib.dump(rf_shap_target_rf, dataset_path+"/RF_shap_"+attack_method+"_on_"+model_type+"_s.pkl")
        joblib.dump(rf_shap_target_gb, dataset_path+"/RF_shap_"+attack_method+"_on_"+model_type+"_s_t.pkl")
    
        
    
    print('training XGBoost models...')
    xgb_shap_target_gb = train_xgb_model(x_gb_shap_train, y_gb_shap_train) 
    xgb_shap_target_rf = train_xgb_model(x_rf_shap_train, y_rf_shap_train) 
           
    print ('saving XGBoost models')
    if (model_type == 'None'):
        joblib.dump(xgb_shap_target_rf, dataset_path+"/XGB_shap_rf_n_np.pkl")
        joblib.dump(xgb_shap_target_gb, dataset_path+"/XGB_shap_gb_n_np.pkl")
    elif(model_type == 'rfgb'):    
        joblib.dump(rf_shap_target_rf, dataset_path+"/XGB_shap_"+attack_method+"_on_rf_s_t_n.pkl")
        joblib.dump(rf_shap_target_gb, dataset_path+"/XGB_shap_"+attack_method+"_on_gb_s_t_n.pkl")
    else:
        joblib.dump(xgb_shap_target_rf, dataset_path+"/XGB_shap_"+attack_method+"_on_"+model_type+"_s.pkl")
        joblib.dump(xgb_shap_target_gb, dataset_path+"/XGB_shap_"+attack_method+"_on_"+model_type+"_s_t.pkl")
        
    
    print('training NLP models...')
    mlp_shap_target_gb = train_mlp_model(x_gb_shap_train, y_gb_shap_train) 
    mlp_shap_target_rf = train_mlp_model(x_rf_shap_train, y_rf_shap_train) 

    print ('saving NLP models')
    if (model_type == 'None'):
        joblib.dump(mlp_shap_target_rf, dataset_path+"/MLP_shap_rf_n_np.pkl")
        joblib.dump(mlp_shap_target_gb, dataset_path+"/MLP_shap_gb_n_np.pkl")
    elif(model_type == 'rfgb'):    
        joblib.dump(rf_shap_target_rf, dataset_path+"/MLP_shap_"+attack_method+"_on_rf_s_t_n.pkl")
        joblib.dump(rf_shap_target_gb, dataset_path+"/MLP_shap_"+attack_method+"_on_gb_s_t_n.pkl")
    else:
        joblib.dump(mlp_shap_target_rf, dataset_path+"/MLP_shap_"+attack_method+"_on_"+model_type+"_s.pkl")
        joblib.dump(mlp_shap_target_gb, dataset_path+"/MLP_shap_"+attack_method+"_on_"+model_type+"_s_t.pkl")
        
    