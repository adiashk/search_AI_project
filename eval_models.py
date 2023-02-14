import sys
import pandas as pd
import numpy as np
import configparser
import random
import pickle
from Utils.models_utils import model_evaluation

import sklearn
from sklearn.utils import resample, shuffle

#from Models.scikitlearn_wrapper import SklearnClassifier
from Utils.anomaly_utils import get_ocsvm
#from Utils.attack_utils import get_hopskipjump, get_constrains
from Utils.data_utils import split_to_datasets, preprocess_ICU
from Utils.models_utils import train_GB_model, train_RF_model
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

def get_config():
    config = configparser.ConfigParser()
    #config.read(sys.argv[1])
    config.read('configurations.txt')
    config = config['DEFAULT']
    return config


if __name__ == '__main__':
    # Set parameters
    configurations = get_config()
    data_path = configurations["data_path"]
    raw_data_path = configurations["raw_data_path"]
    perturbability_path = configurations["perturbability_path"]
    models_path = configurations["models_path"]
    results_path = configurations["results_path"]
    seed = int(configurations["seed"])
    exclude = configurations["exclude"]
    dataset_name = raw_data_path.split("/")[1]

    #preprocess_ICU(raw_data_path)

    # import datasets
    datasets = split_to_datasets(raw_data_path, save_path=data_path)
    
    # import models
    model_type = 'target'

    data_name = data_path.split("/")[-1]
    saving_path = "Models/" 
  
    model_name_GB = "{}_{}_GB_exclude_{}_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type, exclude, seed, 0.1, 10, 3)
    model_name_RF = "{}_{}_RF_exclude_{}_seed-{}_estimators-{}_maxdepth-{}".format(data_name, model_type, exclude,seed, 500, 9)
    GB  = pickle.load(open(models_path + "/" + model_name_GB + ".pkl", 'rb'))
    RF  = pickle.load(open(models_path + "/" + model_name_RF + ".pkl", 'rb'))
    imp_gb = importances = GB.feature_importances_
    imp_rf = importances = RF.feature_importances_

    
    
    # prepare val data
    x_train = datasets["x_train_" + model_type]
    x_val = datasets["x_test"]
    y_train = datasets["y_train_" + model_type]
    y_val = datasets["y_test"]

    features = x_train.columns.to_frame()
    '''
    forest_importances = pd.Series(importances, index=features)
    std = np.std([tree.feature_importances_ for tree in RF.estimators_], axis=0)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
    '''

    for i,v in enumerate(imp_gb):
        print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
    plt.bar([x for x in range(len(imp_gb))], imp_gb)
    plt.show()
    
    for i,v in enumerate(imp_rf):
        print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
    plt.bar([x for x in range(len(imp_rf))], imp_rf)
    plt.show()
    
    features.to_csv(data_path+'/edittible_features.csv', index=False)


    #if ('ICU' in data_name): 
    data_raw = x_val.copy()
    data_raw['pred'] = y_val 
    data_raw = shuffle(data_raw)
    '''
    # under-sampling                                                                                
    g = data_raw.groupby('pred')
    x_val = g.apply(lambda x: x.sample(g.size().max(), replace=True).reset_index(drop=True))
    x_val = data_raw.drop("pred", axis=1)
    y_val = pd.DataFrame(data_raw["pred"])
    '''
    # over-sampling
    # Separate majority and minority classes
    '''
    df_majority = data_raw[data_raw.pred==0]
    df_minority = data_raw[data_raw.pred==1]

    # Resampling the minority levels to match the majority level
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                    replace=True,     # sample with replacement
                                    n_samples=df_majority.shape[0],    # to match majority class
                                    random_state= 303) # reproducible results
    
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled]) #classes are equals now
    y_val = pd.DataFrame(df_upsampled['pred'])
    x_val = df_upsampled.drop('pred', axis=1)
    '''
    
    eval = model_evaluation(model=GB.predict,
                            val_x=x_val,
                            val_y=y_val,
                            #saving_path=saving_path,
                            model_name=model_name)

    # under-sampling                                                                                
    g = data_raw.groupby('pred')
    x_val = g.apply(lambda x: x.sample(g.size().max(), replace=True).reset_index(drop=True))
    x_val = data_raw.drop("pred", axis=1)
    y_val = pd.DataFrame(data_raw["pred"])
    model_name= model_name +'_undersamptest'

    eval_under = model_evaluation(model=GB.predict,
                            val_x=x_val,
                            val_y=y_val,
                            #saving_path=saving_path,
                            model_name=model_name)


    model_name= model_name +'_upsamptest'     

    # eval only class 0
    model_name=model_name+'_0'
    x_val_0 = x_val [y_val.pred == 0].copy()
    y_val_0 = y_val [y_val.pred == 0].copy()
    x_val_1 = x_val [y_val.pred == 1].copy()
    y_val_1 = y_val [y_val.pred == 1].copy()
    eval_0 = model_evaluation(model=GB.predict,
                            val_x=x_val_0,
                            val_y=y_val_0,
                            #saving_path=saving_path,
                            model_name=model_name)

    # eval only class 1
    model_name=model_name+'_1'
    eval_1 = model_evaluation(model=GB.predict,
                            val_x=x_val_1,
                            val_y=y_val_1,
                            #saving_path=saving_path,
                            model_name=model_name)
    print ('s')
