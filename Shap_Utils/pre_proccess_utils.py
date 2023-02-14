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

RANDOM_SEED = config.RANDOM_SEED

def split_x_y(df, pred_col_name):

    y_df = df.copy()[pred_col_name].values
    x_df = df.copy().drop([pred_col_name], axis=1)
    return  x_df, y_df

def prepare_data_for_training(TARGET_col, df_train, df_attack_GB, df_attack_RF, dataset_file, save_path, attack_type):

    if 'boundary' in attack_type:
        name1='gb'
        name2='rf'
    else:
        name1=attack_type+'_s'
        name2=attack_type+'_s_t'

    gb_model, rf_model = get_target_models(dataset_file)
    y_orig_attack = pd.read_csv(dataset_file +'/y_orig_attack.csv').values.flatten()
    #df_train = df_train.drop("Unnamed: 0", axis=1)
    #df_attack_GB = df_attack_GB.drop("Unnamed: 0", axis=1)
    #df_attack_RF = df_attack_RF.drop("Unnamed: 0", axis=1)

    scaler = StandardScaler()
    x_df_train, y_df_train = split_x_y(df_train, TARGET_col)
    scaler = scaler.fit(x_df_train)
    x_df_train[x_df_train.columns] = scaler.transform(x_df_train[x_df_train.columns])

    #select only samples that model predict right
    model_preds_orig_gb = np.where(gb_model.predict(x_df_train) > 0.5, 1, 0) # classified as 1 or 0 according to threshold on original data
    model_preds_orig_rf = np.where(rf_model.predict(x_df_train) > 0.5, 1, 0) # classified as 1 or 0 according to threshold on original data
    true_preds = (model_preds_orig_gb == np.array(y_df_train).reshape(-1, 1).flatten()) * (model_preds_orig_rf == np.array(y_df_train).reshape(-1, 1).flatten()) # filtering sampels that surrogate's prediction was incorrect befor the attack
    x_df_train = x_df_train[true_preds==True]

    df_train = x_df_train.copy()
    df_train['pred'] = y_df_train[true_preds==True].copy()
    
    df_train = shuffle(df_train)
    
    ########### Sub-sampling ###########
    data_0 = df_train[df_train[TARGET_col] == 0]
    data_1 = df_train[df_train[TARGET_col] == 1]

    gb_0 = df_attack_GB[y_orig_attack == 0]
    gb_1 = df_attack_GB[y_orig_attack == 1]

    rf_0 = df_attack_RF[y_orig_attack == 0]
    rf_1 = df_attack_RF[y_orig_attack == 1]

    ########### keep data balanced
    count_labels = min(int(len(data_1)),int(len(data_0)))
    attack_size = len(y_orig_attack)   
    data_0 = data_0[:count_labels]
    data_1 = data_1[:count_labels]
    data_0 = data_0[:int(attack_size/2)]
    data_1 = data_1[:int(attack_size/2)]

    # Keep the complementary examples for later use
    test_size = 0.2
    zeros_train, zeros_test = train_test_split(data_0, test_size=test_size, random_state=RANDOM_SEED)
    ones_train, ones_test = train_test_split(data_1, test_size=test_size, random_state=RANDOM_SEED)

    zeros_train_gb, zeros_test_gb = train_test_split(data_0, test_size=test_size, random_state=RANDOM_SEED)
    ones_train_gb, ones_test_gb = train_test_split(data_1, test_size=test_size, random_state=RANDOM_SEED)

    zeros_train_rf, zeros_test_rf = train_test_split(data_0, test_size=test_size, random_state=RANDOM_SEED)
    ones_train_rf, ones_test_rf = train_test_split(data_1, test_size=test_size, random_state=RANDOM_SEED)

    # prepare df_train 
    x_train = zeros_train.append(ones_train)
    x_train[TARGET_col] = 0

    x_attack_gb = zeros_train_gb.append(ones_train_gb)
    x_attack_gb[TARGET_col] = 1
    x_attack_rf = zeros_train_rf.append(ones_train_rf)
    x_attack_rf[TARGET_col] = 1
    
    df_train_gb = x_train.append(x_attack_gb)
    df_train_rf = x_train.append(x_attack_rf)
    
   # prepare df_test    
    x_test = zeros_test.append(ones_test)
    x_test[TARGET_col] = 0

    x_attack_gb = zeros_test_gb.append(ones_test_gb)
    x_attack_gb[TARGET_col] = 1
    x_attack_rf = zeros_test_rf.append(ones_test_rf)
    x_attack_rf[TARGET_col] = 1
    
    df_test_gb = x_test.append(x_attack_gb)
    df_test_rf = x_test.append(x_attack_rf)
    
    print ('df_train_gb', df_train_gb.shape)
    print ('df_test_gb', df_test_gb.shape)
    print ('df_train_rf', df_train_rf.shape)
    print ('df_test_rf', df_test_rf.shape)
     
    # Save dataset in csv files
    df_train_gb.to_csv(save_path + "/df_train_"+ name1 +".csv", index=False)
    df_test_gb.to_csv(save_path + "/df_test_"+ name1 +".csv", index=False)
    df_train_rf.to_csv(save_path + "/df_train_"+ name2 +".csv", index=False)
    df_test_rf.to_csv(save_path + "/df_test_"+ name2 +".csv", index=False)

    y_train_gb = df_train_gb[TARGET_col]
    y_test_gb = df_test_gb[TARGET_col]
    y_train_rf = df_train_rf[TARGET_col]
    y_test_rf = df_test_rf[TARGET_col] 

    # Remove y column
    x_train_gb, x_test_gb,x_train_rf, x_test_rf = \
        [x.drop([TARGET_col], axis=1) for x in [df_train_gb, df_test_gb, df_train_rf, df_test_rf]]

    return x_train_gb, x_train_rf, x_test_gb, x_test_rf, y_train_gb, y_test_gb,  y_train_rf, y_test_rf

def gen_dataset_for_training(pred_col_name, dataset_file, dataset_train_path, dataset_attack_path, dist, save_path, attack_type):
    if 'boundary' in attack_type:
        df_train = pd.read_csv(dataset_train_path + '/df_sota_train.csv')
        df_attack_GB = pd.read_csv(dataset_attack_path + '/boundary_adv_all_GB_s_' + dist + '.csv')
        df_attack_RF = pd.read_csv(dataset_attack_path + '/boundary_adv_all_RF_s_' + dist + '.csv')
        return prepare_data_for_training(pred_col_name, df_train, df_attack_GB, df_attack_RF, dataset_file, save_path)
    else:
        df_train = pd.read_csv(dataset_train_path + '/df_sota_train.csv')
        df_attack_s = pd.read_csv(dataset_attack_path + '/' + attack_type + '_adv_all_s' + '.csv')
        df_attack_s_t = pd.read_csv(dataset_attack_path + '/' + attack_type + '_adv_all_s_t' + '.csv')
        return prepare_data_for_training(pred_col_name, df_train, df_attack_s, df_attack_s_t, dataset_file, save_path, attack_type)

def get_dataset(pred_col_name, dataset_path, attack_type):

    if 'boundary' in attack_type:
        name1='gb'
        name2='rf'
    else:
        name1=attack_type+'_s'
        name2=attack_type+'_s_t'

    # Load dataset from csv files, we load sota dataset just for printing statistics
    df_train_gb = pd.read_csv(dataset_path + "/df_train_"+ name1 +".csv")
    df_test_gb = pd.read_csv(dataset_path + "/df_test_"+ name1 +".csv")
    df_train_rf= pd.read_csv(dataset_path + "/df_train_"+ name2 +".csv")
    df_test_rf= pd.read_csv(dataset_path + "/df_test_"+ name2 +".csv")

    y_train_gb = df_train_gb[pred_col_name]
    y_test_gb = df_test_gb[pred_col_name]
    y_train_rf = df_train_rf[pred_col_name]
    y_test_rf = df_test_rf[pred_col_name]

    # Remove y column
    x_train_gb, x_test_gb,x_train_rf, x_test_rf = \
        [x.drop([pred_col_name], axis=1) for x in [df_train_gb, df_test_gb, df_train_rf, df_test_rf]]

    return x_train_gb, x_train_rf, x_test_gb, x_test_rf, y_train_gb, y_test_gb,  y_train_rf, y_test_rf

def get_target_models(dataset_file):

    gb_model = joblib.load(dataset_file + "/gb_sota_model.pkl")
    rf_model = joblib.load(dataset_file + "/rf_sota_model.pkl")
    return gb_model, rf_model 

def gen_background_for_shap(dataset_file, target_model):

    df_orig_train = pd.read_csv(dataset_file + "/df_sota_train.csv")
    y_orig_train = df_orig_train['pred']
    x_orig_train = df_orig_train.drop('pred', axis=1)

    #select samples that model predict right
    scaler = StandardScaler()
    x_orig_train[x_orig_train.columns] = scaler.fit_transform(x_orig_train[x_orig_train.columns])

    # Take the train set as a set of background examples to take an expectation over
        #first find most or least confidanced samples
    preds_prob = target_model.predict_proba(x_orig_train)
    x_orig_prob = x_orig_train.copy()
    x_orig_prob["true_y"] = y_orig_train.copy()
    x_orig_prob["prob"] = preds_prob[:,0]

    #select only samples that model predict right
    model_preds_orig = np.where(target_model.predict(x_orig_train) > 0.5, 1, 0) # classified as 1 or 0 according to threshold on original data
    true_preds = model_preds_orig == np.array(y_orig_train).reshape(-1, 1).flatten() # filtering sampels that surrogate's prediction was incorrect befor the attack
    prob = x_orig_prob[true_preds==True]

    prob_0 = prob[prob['true_y'] == 0]
    prob_0 ['prob'] = 1 - prob_0 ['prob']
    prob_0.sort_values(by=["prob"], axis=0, ascending=True, inplace=True)
    prob_0 = prob_0.drop(['prob','true_y'], axis=1)
    
    prob_1 = prob[prob['true_y'] == 1]
    prob_1.sort_values(by=["prob"], axis=0, ascending=True, inplace=True)
    prob_1 = prob_1.drop(['prob','true_y'], axis=1)

    #get high prob
    high_prob_0 = prob_0[:100]
    high_prob_1 = prob_1[:100]
    high_prob = high_prob_0.append(high_prob_1)

    #get low prob
    low_prob_0 = prob_0[-100:]
    low_prob_1 = prob_1[-100:]
    low_prob = low_prob_0.append(low_prob_1)
   
    #get 1 or 0 prob
    prob_1 = low_prob_1.append(high_prob_1).append(high_prob_0).append(low_prob_0)
    prob_0 = high_prob_0.append(low_prob_0)

    if ("HATE" in dataset_file):
        background = x_orig_train[true_preds==True].values
    else:  
        background = high_prob
        
    return background

def gen_shap_dataset(pred_col_name, dataset_file, dataset_path, attack_type):
    
    #TODO take only samples that the models predict correct
    # Get dataset
    print("Loading data for computing shap...")
    x_train_gb, x_train_rf, x_test_gb, x_test_rf, y_train_gb, y_test_gb,  y_train_rf, y_test_rf = \
            get_dataset(pred_col_name, dataset_path, attack_type)
    
    # Reading the SOTA models
    print("Loading target models...")
    gb_target_model, rf_target_model = get_target_models(dataset_file)

    background_gb = gen_background_for_shap(dataset_file, gb_target_model)
    background_rf = gen_background_for_shap(dataset_file, rf_target_model)
    explainer_gb = shap.TreeExplainer(gb_target_model, background_gb)
    explainer_rf = shap.TreeExplainer(rf_target_model, background_rf)
    
    #get_shap_values
    try:
        gb_shap_values = explainer_gb.shap_values(x_train_gb, check_additivity=False)[y_benign[0][0]]
    except:
        print ("error")

    #make train-set of shap values
    gb_shap_train = pd.DataFrame(explainer_gb.shap_values(x_train_gb, check_additivity=False))
    rf_shap_train = pd.DataFrame(explainer_rf.shap_values(x_train_rf, check_additivity=False)[0])

    #gb_shap_train = np.array(gb_shap_train).reshape(-1, x_train_gb.shape[1])
    #rf_shap_train = np.array(rf_shap_train).reshape(-1, x_train_rf.shape[1])

    gb_shap_train['pred'] = y_train_gb
    rf_shap_train['pred'] = y_train_rf

    if 'boundary' not in attack_type:
        gb_shap_train_s_t = pd.DataFrame(explainer_gb.shap_values(x_train_rf, check_additivity=False))
        rf_shap_train_s = pd.DataFrame(explainer_rf.shap_values(x_train_gb, check_additivity=False)[0])
        gb_shap_train_s_t['pred'] = y_train_rf
        rf_shap_train_s['pred'] = y_train_gb 

    #make test-set of shap values
    gb_shap_test = pd.DataFrame(explainer_gb.shap_values(x_test_gb, check_additivity=False))
    rf_shap_test = pd.DataFrame(explainer_rf.shap_values(x_test_rf, check_additivity=False)[0])
    #gb_shap_test = np.array(gb_shap_test).reshape(-1, x_test_gb.shape[1])
    #rf_shap_test = np.array(rf_shap_test).reshape(-1, x_test_rf.shape[1])

    gb_shap_test['pred'] = y_test_gb
    rf_shap_test['pred'] = y_test_rf

    if 'boundary' not in attack_type:
        gb_shap_test_s_t = pd.DataFrame(explainer_gb.shap_values(x_test_rf, check_additivity=False))
        rf_shap_test_s = pd.DataFrame(explainer_rf.shap_values(x_test_gb, check_additivity=False)[0])
        gb_shap_test_s_t['pred'] = y_test_rf
        rf_shap_test_s['pred'] = y_test_gb

     # Save dataset in csv files
    if 'boundary' in attack_type:
        gb_shap_train.to_csv(dataset_path + "/shap_train_gb.csv", index=False)
        rf_shap_train.to_csv(dataset_path + "/shap_train_rf.csv", index=False)
        gb_shap_test.to_csv(dataset_path + "/shap_test_gb.csv", index=False)
        rf_shap_test.to_csv(dataset_path + "/shap_test_rf.csv", index=False)

    else:
        gb_shap_train.to_csv(dataset_path + "/shap_train_on_gb_s.csv", index=False)
        rf_shap_train.to_csv(dataset_path + "/shap_train_on_rf_s_t.csv", index=False)
        gb_shap_test.to_csv(dataset_path + "/shap_test_on_gb_s.csv", index=False)
        rf_shap_test.to_csv(dataset_path + "/shap_test_on_rf_s_t.csv", index=False)
        gb_shap_train_s_t.to_csv(dataset_path + "/shap_train_on_gb_s_t.csv", index=False)
        rf_shap_train_s.to_csv(dataset_path + "/shap_train_on_rf_s.csv", index=False)
        gb_shap_test_s_t.to_csv(dataset_path + "/shap_test_on_gb_s_t.csv", index=False)
        rf_shap_test_s.to_csv(dataset_path + "/shap_test_on_rf_s.csv", index=False)

    #return gb_shap_train, rf_shap_train, gb_shap_test, rf_shap_test
 
def get_shap_dataset(pred_col_name, dataset_file, dataset_path, attack_type):
    if 'boundary' not in attack_type:
        shap_train_gb_s = pd.read_csv(dataset_path + "/shap_train_on_gb_s.csv")
        shap_train_rf_s_t = pd.read_csv(dataset_path + "/shap_train_on_rf_s_t.csv")
        shap_test_gb_s = pd.read_csv(dataset_path + "/shap_test_on_gb_s.csv")
        shap_test_rf_s_t = pd.read_csv(dataset_path + "/shap_test_on_rf_s_t.csv")
        shap_train_gb_s_t = pd.read_csv(dataset_path + "/shap_train_on_gb_s_t.csv")
        shap_train_rf_s = pd.read_csv(dataset_path + "/shap_train_on_rf_s.csv")
        shap_test_gb_s_t = pd.read_csv(dataset_path + "/shap_test_on_gb_s_t.csv")
        shap_test_rf_s = pd.read_csv(dataset_path + "/shap_test_on_rf_s.csv")
        return shap_train_gb_s, shap_train_rf_s_t, shap_test_gb_s, shap_test_rf_s_t, \
            shap_train_gb_s_t, shap_train_rf_s, shap_test_gb_s_t, shap_test_rf_s 
    
    else:
        gb_shap_train = pd.read_csv(dataset_path + "/shap_train_gb.csv")
        rf_shap_train = pd.read_csv(dataset_path + "/shap_train_rf.csv")
        gb_shap_test = pd.read_csv(dataset_path + "/shap_test_gb.csv")
        rf_shap_test = pd.read_csv(dataset_path + "/shap_test_rf.csv")
        return gb_shap_train, rf_shap_train, gb_shap_test, rf_shap_test

def normalaize_shap_dataset(pred_col_name, dataset_file, dataset_path, attack_type):
    if 'boundary' not in attack_type:
        _, shap_train_rf_s_t, _, shap_test_rf_s_t, \
            shap_train_gb_s_t, _, shap_test_gb_s_t, _ = \
                get_shap_dataset(pred_col_name, dataset_file, dataset_path, attack_type)
        
        y_shap_train_rf_s_t, y_shap_test_rf_s_t, y_shap_train_gb_s_t, y_shap_test_gb_s_t = \
                [x.copy()[pred_col_name] for x in [shap_train_rf_s_t, shap_test_rf_s_t, shap_train_gb_s_t, shap_test_gb_s_t]]
        
        x_shap_train_rf_s_t, x_shap_test_rf_s_t, x_shap_train_gb_s_t, x_shap_test_gb_s_t = \
                [x.copy().drop([pred_col_name], axis=1) for x in [shap_train_rf_s_t, shap_test_rf_s_t, shap_train_gb_s_t, shap_test_gb_s_t]]

        scaler = MinMaxScaler()
        x_shap_train_rf_s_t = scaler.fit_transform(x_shap_train_rf_s_t.T).T
        x_shap_train_gb_s_t = scaler.fit_transform(x_shap_train_gb_s_t.T).T
        x_shap_test_rf_s_t = scaler.fit_transform(x_shap_test_rf_s_t.T).T
        x_shap_test_gb_s_t = scaler.fit_transform(x_shap_test_gb_s_t.T).T

        shap_train_rf_s_t[pred_col_name], shap_train_gb_s_t[pred_col_name], shap_test_rf_s_t[pred_col_name], shap_test_gb_s_t[pred_col_name] = \
            y_shap_train_rf_s_t, y_shap_train_gb_s_t, y_shap_test_rf_s_t,  y_shap_test_gb_s_t

        return shap_train_rf_s_t, shap_train_gb_s_t, shap_test_rf_s_t, shap_test_gb_s_t
    
    else:
        gb_shap_train, rf_shap_train, gb_shap_test, rf_shap_test = \
            get_shap_dataset(pred_col_name, dataset_file, dataset_path, attack_type)
    
        y_gb_shap_train, y_rf_shap_train, y_gb_shap_test, y_rf_shap_test = \
                [x.copy()[pred_col_name] for x in [gb_shap_train, rf_shap_train, gb_shap_test, rf_shap_test]]
        
        x_gb_shap_train, x_rf_shap_train, x_gb_shap_test, x_rf_shap_test = [x.copy().drop([pred_col_name], axis=1) for x in [gb_shap_train, rf_shap_train, gb_shap_test, rf_shap_test]]
        
        scaler = MinMaxScaler()
        x_gb_shap_train = scaler.fit_transform(x_gb_shap_train.T).T
        x_rf_shap_train = scaler.fit_transform(x_rf_shap_train.T).T
        x_gb_shap_test = scaler.fit_transform(x_gb_shap_test.T).T
        x_rf_shap_test = scaler.fit_transform(x_rf_shap_test.T).T

        gb_shap_train[pred_col_name], rf_shap_train[pred_col_name], gb_shap_test[pred_col_name], rf_shap_test[pred_col_name] = \
            y_gb_shap_train, y_rf_shap_train, y_gb_shap_test, y_rf_shap_test

        return gb_shap_train, rf_shap_train, gb_shap_test, rf_shap_test


