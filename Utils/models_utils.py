import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import resample, shuffle
from xgboost import XGBClassifier
from sklearn.model_selection  import GridSearchCV


def train_GB_model(data_path, seed=42, val_size=0.2, learning_rate=1.0, n_estimators=100, max_depth=3,
                   saving_path="Models/", datasets=None, model_type="target", exclude=None):
    if datasets is None:
        data_raw = pd.read_csv(data_path + "/df_sota_train.csv")
        data_raw_x = data_raw.drop("pred", axis=1)
        data_raw_y = pd.DataFrame(data_raw["pred"])
        x_train, x_val, y_train, y_val = train_test_split(data_raw_x, data_raw_y, test_size=val_size, random_state=seed)
    else:
        x_train = datasets["x_train_" + model_type]
        x_val = datasets["x_test"]
        y_train = datasets["y_train_" + model_type]
        y_val = datasets["y_test"]

    GB = GradientBoostingClassifier(n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    random_state=seed)

    GB.fit(x_train.to_numpy(), y_train)

    data_name = data_path.split("/")[-1]
    saving_path = saving_path + data_name + "/" 
    if exclude is None:
        model_name = "{}_{}_GB_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type, seed, learning_rate,
                                                                                          n_estimators, max_depth)
    else:
        model_name = "{}_{}_GB_exclude_{}_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type,
                                                                                          exclude, seed, learning_rate,
                                                                                           n_estimators, max_depth)
    pickle.dump(GB, open(saving_path + model_name + ".pkl", 'wb'))                                                                                     

    if ('ICU' in data_name): 
        data_raw = x_val
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
    
    model_name=model_name+'_upsamptest'
    
    eval = model_evaluation(model=GB.predict,
                            val_x=x_val,
                            val_y=y_val,
                            #saving_path=saving_path,
                            model_name=model_name)

    

    return GB, eval


def train_RF_model(data_path, seed=42, val_size=0.2, n_estimators=100, max_depth=3, saving_path="Models/",
                   datasets=None, model_type="target", exclude=None):
    if datasets is None:
        data_raw = pd.read_csv(data_path + "/df_sota_train.csv")
        data_raw_x = data_raw.drop("pred", axis=1)
        data_raw_y = pd.DataFrame(data_raw["pred"])
        x_train, x_val, y_train, y_val = train_test_split(data_raw_x, data_raw_y, test_size=val_size, random_state=seed)
    else:
        x_train = datasets["x_train_" + model_type]
        x_val = datasets["x_test"]
        y_train = datasets["y_train_" + model_type]
        y_val = datasets["y_test"]

    RF = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=seed)
    RF.fit(x_train.to_numpy(), y_train)

    data_name = data_path.split("/")[-1]
    saving_path = saving_path + data_name + "/"

    if exclude is None:
        model_name = "{}_{}_RF_seed-{}_estimators-{}_maxdepth-{}".format(data_name, model_type, seed, n_estimators, max_depth)
    else:
        model_name = "{}_{}_RF_exclude_{}_seed-{}_estimators-{}_maxdepth-{}".format(data_name, model_type,  exclude, seed, n_estimators, max_depth)

    pickle.dump(RF, open(saving_path + model_name + ".pkl", 'wb'))

    if ('ICU' in data_name): 
        data_raw = x_val
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
    
    model_name=model_name+'_upsamptest'

    eval = model_evaluation(model=RF.predict,
                            val_x=x_val,
                            val_y=y_val,
                            #saving_path=saving_path,
                            model_name=model_name)

    

    return RF, eval


def train_XGB_model(data_path, seed=42, val_size=0.2, n_estimators=70, learning_rate=0.1, max_depth=8, saving_path="Models/",
                   datasets=None, model_type="target", exclude=None):
    if datasets is None:
        data_raw = pd.read_csv(data_path + "/df_sota_train.csv")
        data_raw_x = data_raw.drop("pred", axis=1)
        data_raw_y = pd.DataFrame(data_raw["pred"])
        x_train, x_val, y_train, y_val = train_test_split(data_raw_x, data_raw_y, test_size=val_size, random_state=seed)
    else:
        x_train = datasets["x_train_" + model_type]
        x_val = datasets["x_test"]
        y_train = datasets["y_train_" + model_type]
        y_val = datasets["y_test"]

    estimator = XGBClassifier(objective= 'binary:hinge', nthread=4, seed=seed,
                                                                    n_estimators = n_estimators,
                                                                    learning_rate = learning_rate,
                                                                    max_depth = max_depth, 
                                                                    use_label_encoder=False)
    #parameters = {'max_depth': range (1, 10, 1), 'n_estimators': range(5, 100, 5), 'learning_rate': [0.5, 0.1, 0.01, 0.05]}
    '''
    if( "HATE" in dataset_file):
        param_test1 = {'n_estimators': range(20, 100, 10), 'max_depth': range(1, 5, 1)}#10
    else:
        param_test1 = {'n_estimators': range(20, 1000, 10), 'max_depth': range(1, 5, 1)}#30
    '''
    '''
    gsearch = GridSearchCV(
        estimator = estimator, param_grid=parameters, scoring='accuracy',  n_jobs=-1,  cv=5)
    
    #gsearch = GridSearchCV(
    #    estimator=HistGradientBoostingClassifier(learning_rate=0.1, random_state=1234),
    #    param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    
    #print("XGB: performing hyperparameter tuning using GridSearchCV, cv=5...")
    gsearch.fit(x_train.to_numpy(), y_train)

    print("Best params found: " + str(gsearch.best_params_)) #0.1 70 8
    gsearch.best_estimator_.fit(x_train, y_train)
    XGB = gsearch.best_estimator_
    '''
    XGB = estimator.fit(x_train.to_numpy(), y_train)
    data_name = data_path.split("/")[-1]
    saving_path = saving_path + data_name + "/"

    if exclude is None:
        model_name = "{}_{}_XGB_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type, seed, learning_rate, n_estimators, max_depth)
    else:
        model_name = "{}_{}_XGB_exclude_{}_seed-{}_lr-{}_estimators-{}_maxdepth-{}".format(data_name, model_type,  exclude, seed, learning_rate, n_estimators, max_depth)

    pickle.dump(XGB, open(saving_path + model_name + ".pkl", 'wb'))

    if ('ICU' in data_name): 
        data_raw = x_val
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
    
    model_name=model_name+'_upsamptest'

    eval = model_evaluation(model=XGB.predict,
                            val_x=x_val,
                            val_y=y_val,
                            #saving_path=saving_path,
                            model_name=model_name)

    

    return XGB, eval

def model_evaluation(model, val_x, val_y, saving_path="Models/", model_name="model"):
    eval = {
        "accuracy_score": [accuracy_score(val_y, model(val_x))],
        "f1_score": [f1_score(val_y, model(val_x))],
        "precision_score": [precision_score(val_y, model(val_x))],
        "recall_score": [recall_score(val_y, model(val_x))]
    }
    pd.DataFrame(eval).to_csv(saving_path + "Evaluation/" + model_name + "_no_year.csv", index=False)
    return eval


# pickle.load(open(filename, 'rb'))