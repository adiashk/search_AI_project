import os
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
import copy
from sklearn.utils import resample
# Import IterativeImputer from fancyimpute
#from fancyimpute import IterativeImputer


def split_to_datasets(raw_data_path, seed=42, val_size=0.25, surrgate_train_size=0.5, save_path=None, exclude=None):
    files = os.listdir(save_path)
    if "x_train_target_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size) in files:
        x_train_target = pd.read_csv(save_path + "/x_train_target_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size))
        x_train_surrogate = pd.read_csv(save_path + "/x_train_surrogate_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size))
        y_train_target = pd.read_csv(save_path + "/y_train_target_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size))
        y_train_surrogate = pd.read_csv(save_path + "/y_train_surrogate_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size))
        x_test = pd.read_csv(save_path + "/x_test_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size))
        y_test = pd.read_csv(save_path + "/y_test_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            exclude,seed, val_size, surrgate_train_size))
        datasets = {
            "x_train_target": x_train_target,
            "x_train_surrogate": x_train_surrogate,
            "y_train_target": y_train_target,
            "y_train_surrogate": y_train_surrogate,
            "x_test": x_test,
            "y_test": y_test
        }
        return datasets

    if "x_train_target_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            seed, val_size, surrgate_train_size) in files and exclude is None:
        x_train_target = pd.read_csv(save_path + "/x_train_target_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            seed, val_size, surrgate_train_size))
        x_train_surrogate = pd.read_csv(save_path +
            "/x_train_surrogate_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
                seed, val_size, surrgate_train_size))
        y_train_target = pd.read_csv(save_path + "/y_train_target_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            seed, val_size, surrgate_train_size))
        y_train_surrogate = pd.read_csv(save_path +
            "/y_train_surrogate_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
                seed, val_size, surrgate_train_size))
        x_test = pd.read_csv(save_path + "/x_test_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            seed, val_size, surrgate_train_size))
        y_test = pd.read_csv(save_path + "/y_test_seed_{}_val_size_{}_surrgate_train_size_{}.csv".format(
            seed, val_size, surrgate_train_size))
        datasets = {
            "x_train_target": x_train_target,
            "x_train_surrogate": x_train_surrogate,
            "y_train_target": y_train_target,
            "y_train_surrogate": y_train_surrogate,
            "x_test": x_test,
            "y_test": y_test
        }
        return datasets
  
    data_raw = pd.read_csv(raw_data_path+'/after_preprocessing/RADCOM_after_preprocessing_no_year.csv').reset_index(drop=True)
    if "Unnamed: 0" in data_raw.columns:
        data_raw = data_raw.drop(["Unnamed: 0"], axis=1)

    if raw_data_path.split("/")[1] == "HATE":
        data_raw["hate_neigh"] = data_raw["hate_neigh"].apply(lambda x: int(x))

    if exclude is not None:
        data_raw = data_raw.drop(exclude, axis=1)

    pred0 = len(data_raw[data_raw.pred==0])
    pred1 = len(data_raw[data_raw.pred==1])
    # balance the data
    if raw_data_path.split("/")[1] == "HATE":
        g = data_raw.groupby('pred')
        data_raw = g.apply(lambda x: x.sample(g.size().max(), replace=True).reset_index(drop=True))

        data_raw_x = data_raw.drop("pred", axis=1)
        data_raw_y = pd.DataFrame(data_raw["pred"])
        x_train, x_test, y_train, y_test = train_test_split(data_raw_x,
                                                            data_raw_y,
                                                            test_size=val_size,
                                                            random_state=seed)
        x_train_target, x_train_surrogate, y_train_target, y_train_surrogate = train_test_split(x_train,
                                                                                                y_train,
                                                                                                 random_state=seed)

                                         
    if [(x == raw_data_path.split("/")[1]) for x in ["ICU", "RADCOM"]]:
    #for [x in ["ICU", "RADCOM"] if (x = raw_data_path.split("/")[1])]
        # Separate majority and minority classes
        df_majority = data_raw[data_raw.pred==0]
        df_minority = data_raw[data_raw.pred==1]

        Train_maj, Test_maj = train_test_split(df_majority, test_size = val_size, random_state = seed)
        Train_min, Test_min = train_test_split(df_minority, test_size = val_size, random_state = seed)


        # Resampling the minority levels to match the majority level
        # Upsample minority class
        df_minority_upsampled = resample(Train_min, 
                                        replace=True,     # sample with replacement
                                        n_samples=Train_maj.shape[0],    # to match majority class
                                        random_state= 303) # reproducible results
        
        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([Train_maj, df_minority_upsampled]) #classes are equals now
        test =  pd.concat([Test_maj, Test_min])

        train_target, train_surrogate = train_test_split(df_upsampled, 
                                                        test_size = surrgate_train_size,
                                                        shuffle=True, 
                                                        random_state = seed)

        if (raw_data_path.split("/")[1] == "RADCOM"):
            x_train_target = train_target.copy().drop('pred', axis = 1)
            y_train_target = train_target[['pred']]
            x_train_surrogate = train_surrogate.copy().drop('pred', axis = 1)
            y_train_surrogate = train_surrogate[['pred']]
            x_test = test.copy().drop('pred', axis = 1)
            y_test = test[['pred']]

        else:
            x_train_target = train_target.copy().drop('pred', axis = 1)
            y_train_target = train_target[['encounter_id','pred']]
            x_train_surrogate = train_surrogate.copy().drop('pred', axis = 1)
            y_train_surrogate = train_surrogate[['encounter_id','pred']]
            x_test = test.copy().drop('pred', axis = 1)
            y_test = test[['encounter_id', 'pred']]

            x_train_target.set_index('encounter_id', inplace = True)
            y_train_target.set_index('encounter_id', inplace = True)
            x_train_surrogate.set_index('encounter_id', inplace = True)
            y_train_surrogate.set_index('encounter_id', inplace = True)
            x_test.set_index('encounter_id', inplace = True)
            y_test.set_index('encounter_id', inplace = True)
                                                                                                                     
                                                                                               
    datasets = {
        "x_train_target": x_train_target,
        "x_train_surrogate": x_train_surrogate,
        "y_train_target": y_train_target,
        "y_train_surrogate": y_train_surrogate,
        "x_test": x_test,
        "y_test": y_test
    }

    if save_path is not None:
        for key in datasets.keys():
            if exclude is not None:
                file_name = str(key) + "_exclude_{}_seed_{}_val_size_{}_surrgate_train_size_{}_no_year.csv".format(exclude,seed, val_size, surrgate_train_size)
            else:
                file_name = str(key) + "_seed_{}_val_size_{}_surrgate_train_size_{}_no_year.csv".format(seed, val_size, surrgate_train_size)
            cur_saving_path = save_path + "/" + file_name
            datasets.get(key).to_csv(cur_saving_path, index=False)

    # save edittitible features
    features = x_train_target.columns.to_frame()
    features.to_csv(save_path+'/edittible_features_no_year.csv', index=False)

    return datasets


# ___________________________________________________CREDIT________________________________________________________

def get_age_group(days_birth):
    age_years = -days_birth / 365
    if age_years < 27:
        return 1
    elif age_years < 40:
        return 2
    elif age_years < 50:
        return 3
    elif age_years < 65:
        return 4
    elif age_years < 99:
        return 5
    else:
        return 0


def do_mean(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return df


def do_median(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].median().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return df


def do_std(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].std().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return df


def do_sum(df, group_cols, counted, agg_name):
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return df


def label_encoder(df, categorical_columns=None):
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])
    return df, categorical_columns


def drop_application_columns(df):
    drop_list = [
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START',
        'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'FLAG_PHONE',
        'FLAG_OWN_REALTY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_WORK_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR',
        'COMMONAREA_MODE', 'NONLIVINGAREA_MODE', 'ELEVATORS_MODE', 'NONLIVINGAREA_AVG',
        'FLOORSMIN_MEDI', 'LANDAREA_MODE', 'NONLIVINGAREA_MEDI', 'LIVINGAPARTMENTS_MODE',
        'FLOORSMIN_AVG', 'LANDAREA_AVG', 'FLOORSMIN_MODE', 'LANDAREA_MEDI',
        'COMMONAREA_MEDI', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'BASEMENTAREA_AVG',
        'BASEMENTAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
        'LIVINGAPARTMENTS_AVG', 'ELEVATORS_AVG', 'YEARS_BUILD_MEDI', 'ENTRANCES_MODE',
        'NONLIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MEDI',
        'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_MEDI', 'LIVINGAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MODE', 'NONLIVINGAPARTMENTS_AVG', 'HOUSETYPE_MODE',
        'FONDKAPREMONT_MODE', 'EMERGENCYSTATE_MODE'
    ]
    for doc_num in [2,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,21]:
        drop_list.append('FLAG_DOCUMENT_{}'.format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    return df


def one_hot_encoder(df, categorical_columns=None, nan_as_category=True):
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns


def group_f(df_to_agg, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()


def group_and_merge(df_to_agg, df_to_merge, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    agg_df = group_f(df_to_agg, prefix, aggregations, aggregate_by= aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on= aggregate_by)


def get_bureau_balance(path, num_rows= None):
    bb = pd.read_csv(os.path.join(path, 'bureau_balance.csv'))
    bb, categorical_cols = one_hot_encoder(bb, nan_as_category= False)
    # Calculate rate for each category with decay
    bb_processed = bb.groupby('SK_ID_BUREAU')[categorical_cols].mean().reset_index()
    # Min, Max, Count and mean duration of payments (months)
    agg = {'MONTHS_BALANCE': ['min', 'max', 'mean', 'size']}
    bb_processed = group_and_merge(bb, bb_processed, '', agg, 'SK_ID_BUREAU')
    return bb_processed


def preprocess_CREDIT(init_data_path):
    # Taken from https://www.kaggle.com/code/lolokiller/model-with-bayesian-optimization
    # Addition: remove highly correlated features (correlation higher then 90%) and not important features.

    df = pd.read_csv(init_data_path + "/application_train.csv")

    # Remove outliers
    df = df[df['AMT_INCOME_TOTAL'] < 20000000]
    df = df[df['CODE_GENDER'] != 'XNA']
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    # Feature engineering
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    df['NEW_DOC_KURT'] = df[docs].kurtosis(axis=1)
    df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_group(x))
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3
    np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
        feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
        df[feature_name] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    group = ['ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'AGE_RANGE', 'CODE_GENDER']
    df = do_median(df, group, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_MEDIAN')
    df = do_std(df, group, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_STD')
    df = do_mean(df, group, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_MEAN')
    df = do_std(df, group, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_STD')
    df = do_mean(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_MEAN')
    df = do_std(df, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_STD')
    df = do_mean(df, group, 'AMT_CREDIT', 'GROUP_CREDIT_MEAN')
    df = do_mean(df, group, 'AMT_ANNUITY', 'GROUP_ANNUITY_MEAN')
    df = do_std(df, group, 'AMT_ANNUITY', 'GROUP_ANNUITY_STD')

    # Encoding categorical
    df, le_encoded_cols = label_encoder(df, None)
    df = drop_application_columns(df)
    df = pd.get_dummies(df)

    # Add Bureau data
    bureau = pd.read_csv(init_data_path + "/bureau.csv")
    bureau['CREDIT_DURATION'] = -bureau['DAYS_CREDIT'] + bureau['DAYS_CREDIT_ENDDATE']
    bureau['ENDDATE_DIF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    bureau['DEBT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
    bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    bureau['CREDIT_TO_ANNUITY_RATIO'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']
    bureau, categorical_cols = one_hot_encoder(bureau, nan_as_category=False)
    bureau = bureau.merge(get_bureau_balance(init_data_path), how='left', on='SK_ID_BUREAU')
    bureau['STATUS_12345'] = 0
    for i in range(1, 6):
        bureau['STATUS_12345'] += bureau['STATUS_{}'.format(i)]

    features = ['AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_SUM',
                'AMT_CREDIT_SUM_DEBT', 'DEBT_PERCENTAGE', 'DEBT_CREDIT_DIFF', 'STATUS_0', 'STATUS_12345']
    agg_length = bureau.groupby('MONTHS_BALANCE_SIZE')[features].mean().reset_index()
    agg_length.rename({feat: 'LL_' + feat for feat in features}, axis=1, inplace=True)
    bureau = bureau.merge(agg_length, how='left', on='MONTHS_BALANCE_SIZE')
    BUREAU_AGG = {
        'SK_ID_BUREAU': ['nunique'],
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['mean'],
        'DEBT_CREDIT_DIFF': ['mean', 'sum'],
        'MONTHS_BALANCE_MEAN': ['mean', 'var'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'STATUS_0': ['mean'],
        'STATUS_1': ['mean'],
        'STATUS_12345': ['mean'],
        'STATUS_C': ['mean'],
        'STATUS_X': ['mean'],
        'CREDIT_ACTIVE_Active': ['mean'],
        'CREDIT_ACTIVE_Closed': ['mean'],
        'CREDIT_ACTIVE_Sold': ['mean'],
        'CREDIT_TYPE_Consumer credit': ['mean'],
        'CREDIT_TYPE_Credit card': ['mean'],
        'CREDIT_TYPE_Car loan': ['mean'],
        'CREDIT_TYPE_Mortgage': ['mean'],
        'CREDIT_TYPE_Microloan': ['mean'],
        'LL_AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'LL_DEBT_CREDIT_DIFF': ['mean'],
        'LL_STATUS_12345': ['mean'],
    }
    BUREAU_ACTIVE_AGG = {
        'DAYS_CREDIT': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['min', 'mean'],
        'DEBT_PERCENTAGE': ['mean'],
        'DEBT_CREDIT_DIFF': ['mean'],
        'CREDIT_TO_ANNUITY_RATIO': ['mean'],
        'MONTHS_BALANCE_MEAN': ['mean', 'var'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
    }
    BUREAU_CLOSED_AGG = {
        'DAYS_CREDIT': ['max', 'var'],
        'DAYS_CREDIT_ENDDATE': ['max'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'sum'],
        'DAYS_CREDIT_UPDATE': ['max'],
        'ENDDATE_DIF': ['mean'],
        'STATUS_12345': ['mean'],
    }
    BUREAU_LOAN_TYPE_AGG = {
        'DAYS_CREDIT': ['mean', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM': ['mean', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'max'],
        'DEBT_PERCENTAGE': ['mean'],
        'DEBT_CREDIT_DIFF': ['mean'],
        'DAYS_CREDIT_ENDDATE': ['max'],
    }
    BUREAU_TIME_AGG = {
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'DEBT_PERCENTAGE': ['mean'],
        'DEBT_CREDIT_DIFF': ['mean'],
        'STATUS_0': ['mean'],
        'STATUS_12345': ['mean'],
    }
    agg_bureau = group_f(bureau, 'BUREAU_', BUREAU_AGG)
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    agg_bureau = group_and_merge(active, agg_bureau, 'BUREAU_ACTIVE_', BUREAU_ACTIVE_AGG)
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    agg_bureau = group_and_merge(closed, agg_bureau, 'BUREAU_CLOSED_', BUREAU_CLOSED_AGG)
    for credit_type in ['Consumer credit', 'Credit card', 'Mortgage', 'Car loan', 'Microloan']:
        type_df = bureau[bureau['CREDIT_TYPE_' + credit_type] == 1]
        prefix = 'BUREAU_' + credit_type.split(' ')[0].upper() + '_'
        agg_bureau = group_and_merge(type_df, agg_bureau, prefix, BUREAU_LOAN_TYPE_AGG)
    for time_frame in [6, 12]:
        prefix = "BUREAU_LAST{}M_".format(time_frame)
        time_frame_df = bureau[bureau['DAYS_CREDIT'] >= -30 * time_frame]
        agg_bureau = group_and_merge(time_frame_df, agg_bureau, prefix, BUREAU_TIME_AGG)
    sort_bureau = bureau.sort_values(by=['DAYS_CREDIT'])
    gr = sort_bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].last().reset_index()
    gr.rename({'AMT_CREDIT_MAX_OVERDUE': 'BUREAU_LAST_LOAN_MAX_OVERDUE'}, inplace=True)
    agg_bureau = agg_bureau.merge(gr, on='SK_ID_CURR', how='left')
    agg_bureau['BUREAU_DEBT_OVER_CREDIT'] = \
        agg_bureau['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / agg_bureau['BUREAU_AMT_CREDIT_SUM_SUM']
    agg_bureau['BUREAU_ACTIVE_DEBT_OVER_CREDIT'] = \
        agg_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM'] / agg_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM']
    df = pd.merge(df, agg_bureau, on='SK_ID_CURR', how='left')

    # Adding Installments/ previous application data
    prev = pd.read_csv(os.path.join(init_data_path, 'previous_application.csv'))
    pay = pd.read_csv(os.path.join(init_data_path, 'installments_payments.csv'))
    PREVIOUS_AGG = {
        'SK_ID_PREV': ['nunique'],
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['max', 'mean'],
        'DAYS_TERMINATION': ['max'],
        # Engineered features
        'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
        'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean'],
        'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean', 'var'],
        'DOWN_PAYMENT_TO_CREDIT': ['mean'],
    }
    PREVIOUS_ACTIVE_AGG = {
        'SK_ID_PREV': ['nunique'],
        'SIMPLE_INTERESTS': ['mean'],
        'AMT_ANNUITY': ['max', 'sum'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['sum'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['min', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        # Engineered features
        'AMT_PAYMENT': ['sum'],
        'INSTALMENT_PAYMENT_DIFF': ['mean', 'max'],
        'REMAINING_DEBT': ['max', 'mean', 'sum'],
        'REPAYMENT_RATIO': ['mean'],
    }
    PREVIOUS_LATE_PAYMENTS_AGG = {
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        # Engineered features
        'APPLICATION_CREDIT_DIFF': ['min'],
        'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
        'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
        'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
    }
    PREVIOUS_LOAN_TYPE_AGG = {
        'AMT_CREDIT': ['sum'],
        'AMT_ANNUITY': ['mean', 'max'],
        'SIMPLE_INTERESTS': ['min', 'mean', 'max', 'var'],
        'APPLICATION_CREDIT_DIFF': ['min', 'var'],
        'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['max'],
        'DAYS_LAST_DUE_1ST_VERSION': ['max', 'mean'],
        'CNT_PAYMENT': ['mean'],
    }
    PREVIOUS_TIME_AGG = {
        'AMT_CREDIT': ['sum'],
        'AMT_ANNUITY': ['mean', 'max'],
        'SIMPLE_INTERESTS': ['mean', 'max'],
        'DAYS_DECISION': ['min', 'mean'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        # Engineered features
        'APPLICATION_CREDIT_DIFF': ['min'],
        'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
        'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
        'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
        'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
    }
    PREVIOUS_APPROVED_AGG = {
        'SK_ID_PREV': ['nunique'],
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['max'],
        'AMT_GOODS_PRICE': ['max'],
        'HOUR_APPR_PROCESS_START': ['min', 'max'],
        'DAYS_DECISION': ['min', 'mean'],
        'CNT_PAYMENT': ['max', 'mean'],
        'DAYS_TERMINATION': ['mean'],
        # Engineered features
        'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
        'APPLICATION_CREDIT_DIFF': ['max'],
        'APPLICATION_CREDIT_RATIO': ['min', 'max', 'mean'],
        # The following features are only for approved applications
        'DAYS_FIRST_DRAWING': ['max', 'mean'],
        'DAYS_FIRST_DUE': ['min', 'mean'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE': ['max', 'mean'],
        'DAYS_LAST_DUE_DIFF': ['min', 'max', 'mean'],
        'SIMPLE_INTERESTS': ['min', 'max', 'mean'],
    }
    PREVIOUS_REFUSED_AGG = {
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['min', 'max'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['max', 'mean'],
        # Engineered features
        'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean', 'var'],
        'APPLICATION_CREDIT_RATIO': ['min', 'mean'],
        'NAME_CONTRACT_TYPE_Consumer loans': ['mean'],
        'NAME_CONTRACT_TYPE_Cash loans': ['mean'],
        'NAME_CONTRACT_TYPE_Revolving loans': ['mean'],
    }
    ohe_columns = [
        'NAME_CONTRACT_STATUS', 'NAME_CONTRACT_TYPE', 'CHANNEL_TYPE',
        'NAME_TYPE_SUITE', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',
        'NAME_PRODUCT_TYPE', 'NAME_CLIENT_TYPE']
    prev, categorical_cols = one_hot_encoder(prev, ohe_columns, nan_as_category=False)
    prev['APPLICATION_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['APPLICATION_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
    prev['DOWN_PAYMENT_TO_CREDIT'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']
    total_payment = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    prev['SIMPLE_INTERESTS'] = (total_payment / prev['AMT_CREDIT'] - 1) / prev['CNT_PAYMENT']

    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    active_df = approved[approved['DAYS_LAST_DUE'] == 365243]
    active_pay = pay[pay['SK_ID_PREV'].isin(active_df['SK_ID_PREV'])]
    active_pay_agg = active_pay.groupby('SK_ID_PREV')[['AMT_INSTALMENT', 'AMT_PAYMENT']].sum()
    active_pay_agg.reset_index(inplace=True)
    active_pay_agg['INSTALMENT_PAYMENT_DIFF'] = active_pay_agg['AMT_INSTALMENT'] - active_pay_agg['AMT_PAYMENT']
    active_df = active_df.merge(active_pay_agg, on='SK_ID_PREV', how='left')
    active_df['REMAINING_DEBT'] = active_df['AMT_CREDIT'] - active_df['AMT_PAYMENT']
    active_df['REPAYMENT_RATIO'] = active_df['AMT_PAYMENT'] / active_df['AMT_CREDIT']
    active_agg_df = group_f(active_df, 'PREV_ACTIVE_', PREVIOUS_ACTIVE_AGG)
    active_agg_df['TOTAL_REPAYMENT_RATIO'] = active_agg_df['PREV_ACTIVE_AMT_PAYMENT_SUM'] / \
                                             active_agg_df['PREV_ACTIVE_AMT_CREDIT_SUM']
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    prev['DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']
    approved['DAYS_LAST_DUE_DIFF'] = approved['DAYS_LAST_DUE_1ST_VERSION'] - approved['DAYS_LAST_DUE']

    categorical_agg = {key: ['mean'] for key in categorical_cols}

    agg_prev = group_f(prev, 'PREV_', {**PREVIOUS_AGG, **categorical_agg})
    agg_prev = agg_prev.merge(active_agg_df, how='left', on='SK_ID_CURR')
    agg_prev = group_and_merge(approved, agg_prev, 'APPROVED_', PREVIOUS_APPROVED_AGG)
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    agg_prev = group_and_merge(refused, agg_prev, 'REFUSED_', PREVIOUS_REFUSED_AGG)
    for loan_type in ['Consumer loans', 'Cash loans']:
        type_df = prev[prev['NAME_CONTRACT_TYPE_{}'.format(loan_type)] == 1]
        prefix = 'PREV_' + loan_type.split(" ")[0] + '_'
        agg_prev = group_and_merge(type_df, agg_prev, prefix, PREVIOUS_LOAN_TYPE_AGG)
    pay['LATE_PAYMENT'] = pay['DAYS_ENTRY_PAYMENT'] - pay['DAYS_INSTALMENT']
    pay['LATE_PAYMENT'] = pay['LATE_PAYMENT'].apply(lambda x: 1 if x > 0 else 0)
    dpd_id = pay[pay['LATE_PAYMENT'] > 0]['SK_ID_PREV'].unique()

    agg_dpd = group_and_merge(prev[prev['SK_ID_PREV'].isin(dpd_id)], agg_prev,
                              'PREV_LATE_', PREVIOUS_LATE_PAYMENTS_AGG)
    for time_frame in [12, 24]:
        time_frame_df = prev[prev['DAYS_DECISION'] >= -30 * time_frame]
        prefix = 'PREV_LAST{}M_'.format(time_frame)
        agg_prev = group_and_merge(time_frame_df, agg_prev, prefix, PREVIOUS_TIME_AGG)
    df = pd.merge(df, agg_prev, on='SK_ID_CURR', how='left')

    # Fill NA and scale
    labels = df['TARGET']
    df = df.drop(columns=["TARGET"])
    colonnes = df.columns
    feature = list(df.columns)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='median')
    imputer.fit(df)
    df = imputer.transform(df)
    pickle.dump(imputer, open(init_data_path + "/imputer.pkl", 'wb'))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df)
    df = scaler.transform(df)
    pickle.dump(scaler, open(init_data_path + "/scaler.pkl", 'wb'))

    x = pd.DataFrame(df, columns=colonnes)
    varthresh = VarianceThreshold(threshold=x.var().describe()[4])  # 25% lowest variance threshold
    x = varthresh.fit_transform(x)
    pickle.dump(varthresh, open(init_data_path + "/varthresh.pkl", 'wb'))

    feature_after_vartresh = varthresh.get_feature_names_out(feature)
    final = pd.DataFrame(x, columns=feature_after_vartresh)

    final = drop_corolated_all(final)

    # drop not important
    drop_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/exclude.csv")["feature"].values.tolist()
    final = final.drop(drop_list, axis=1)

    # take top 150
    top_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/top_150.csv")[
        "feature"].values.tolist()
    final = final[top_list]

    final["pred"] = labels
    final.to_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/credit_after_preprocessing.csv",
                 index=False)


def drop_corolated_all(df):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.9
    to_drop_90 = [column for column in upper.columns if any(upper[column] > 0.9)]

    return df.drop(to_drop_90, axis=1)

# ___________________________________________________ICU________________________________________________________

def preprocess_ICU_(init_data_path):
    # Taken from https://www.kaggle.com/code/binaicrai/fork-of-fork-of-wids-lgbm-gs
    # Addition: remove highly correlated features (correlation higher then 90%) and not important features.

    train = pd.read_csv(init_data_path + "/original/training_v2.csv")
    test = pd.read_csv(init_data_path + "/original/unlabeled.csv")
    train_len = len(train)
    combined_dataset = pd.concat(objs = [train, test], axis = 0)

    # cleanlab works with **any classifier**. Yup, you can use sklearn/PyTorch/TensorFlow/XGBoost/etc.
    cl = cleanlab.classification.CleanLearning(sklearn.YourFavoriteClassifier())

    # cleanlab finds data and label issues in **any dataset**... in ONE line of code!
    label_issues = cl.find_label_issues(data, labels)

    # cleanlab trains a robust version of your model that works more reliably with noisy data.
    cl.fit(data, labels)

    # cleanlab estimates the predictions you would have gotten if you had trained with *no* label issues.
    cl.predict(test_data)

    # A true data-centric AI package, cleanlab quantifies class-level issues and overall data quality, for any dataset.
    cleanlab.dataset.health_summary(labels, confident_joint=cl.confident_joint)

def preprocess_ICU(init_data_path):
    # Taken from https://www.kaggle.com/code/binaicrai/fork-of-fork-of-wids-lgbm-gs
    # Addition: remove highly correlated features (correlation higher then 90%) and not important features.

    train = pd.read_csv(init_data_path + "/original/training_v2.csv")
    test = pd.read_csv(init_data_path + "/original/unlabeled.csv")
    train_len = len(train)
    combined_dataset = pd.concat(objs = [train, test], axis = 0)

    # Extracing categorical columns
    
    #df_cat = combined_dataset.select_dtypes(include=['object', 'category']) 

    '''
    'hospital_admit_source':
    Grouping: ['Other ICU', 'ICU']; ['ICU to SDU', 'Step-Down Unit (SDU)']; ['Other Hospital', 'Other']; ['Recovery Room','Observatoin']
    Renaming: Acute Care/Floor to Acute Care
    'icu_type':
    Grouping of the following can be explored: ['CCU-CTICU', 'CTICU', 'Cardiac ICU']
    'apache_2_bodysystem':
    Grouping of the following can be explored: ['Undefined Diagnoses', 'Undefined diagnoses']
    '''
    combined_dataset['hospital_admit_source'] = combined_dataset['hospital_admit_source'].replace({'Other ICU': 'ICU','ICU to SDU':'SDU', 'Step-Down Unit (SDU)': 'SDU',
                                                                                               'Other Hospital':'Other','Observation': 'Recovery Room','Acute Care/Floor': 'Acute Care'})
    # combined_dataset['icu_type'] = combined_dataset['icu_type'].replace({'CCU-CTICU': 'Grpd_CICU', 'CTICU':'Grpd_CICU', 'Cardiac ICU':'Grpd_CICU'}) # Can be explored
    combined_dataset['apache_2_bodysystem'] = combined_dataset['apache_2_bodysystem'].replace({'Undefined diagnoses': 'Undefined Diagnoses'})   
    
    # Dropping few column/s with single value and all unique values
    # Dropping 'readmission_status', 'patient_id', along with 'gender'
    combined_dataset = combined_dataset.drop(['readmission_status', 'patient_id', 'gender'], axis=1)
    
    train = copy.copy(combined_dataset[:train_len])
    test = copy.copy(combined_dataset[train_len:])

    # Checking NAs for initial column clipping 
    # On train data
    pd.set_option('display.max_rows', 500)
    NA_col_train = pd.DataFrame(train.isna().sum(), columns = ['NA_Count'])
    NA_col_train['% of NA'] = (NA_col_train.NA_Count/len(train))*100
    NA_col_train.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')
    
    # Setting threshold of 80%
    NA_col_train = NA_col_train[NA_col_train['% of NA'] >= 80]
    cols_to_drop = NA_col_train.index.tolist()
    # cols_to_drop.remove('hospital_death')
   
    # Dropping columns with >= 80% of NAs
    combined_dataset = combined_dataset.drop(cols_to_drop, axis=1)
    
    train = copy.copy(combined_dataset[:train_len])
    test = copy.copy(combined_dataset[train_len:])

    # MICE Imputation
    #7. MICE Imputation 
    #['hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem']
    # Suggestion Courtesy: Bruno Taveres - https://www.kaggle.com/c/widsdatathon2020/discussion/130532
    # Adding 2 apache columns as well


    # Initialize IterativeImputer
    mice_imputer = IterativeImputer()

    # Impute using fit_tranform on diabetes
    train[['age', 'height', 'weight', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']] = mice_imputer.fit_transform(train[['age', 'height', 'weight', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']])
    
    # impute BMI = Weight(kg)/(Height(m)* Height(m))
    train['new_bmi'] = (train['weight']*10000)/(train['height']*train['height'])
    train['bmi'] = train['bmi'].fillna(train['new_bmi'])
    train = train.drop(['new_bmi'], axis = 1)

    # Extracting columns to change to Categorical
    col_train = train.columns
    l1 = []
    for i in col_train:
        if train[i].nunique() <= 16:
            l1.append(i)
                
    l1.remove('hospital_death')
    train[l1] = train[l1].apply(lambda x: x.astype('category'), axis=0)

    cols = train.columns
    num_cols = train._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))

    # Fill NA
    df=train.copy()
    labels = df['hospital_death']
    df = df.drop(columns=["hospital_death"])
    colonnes = df.columns
    feature = list(df.columns)

    df.replace([np.inf, -np.inf,'NA'], np.nan, inplace=True)
    
    imputer_numeric = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    numeric_cols= list(set(num_cols) - set(['hospital_death']))
    imputer_numeric.fit(df[numeric_cols])
    imputer_cat.fit(df[cat_cols])
    df[cat_cols] = imputer_cat.transform(df[cat_cols])
    df[numeric_cols] = imputer_numeric.transform(df[numeric_cols])
    #pickle.dump(imputer, open(init_data_path + "/imputer.pkl", 'wb'))
    pickle.dump(imputer_cat, open(init_data_path + "/imputer_cat.pkl", 'wb'))
    pickle.dump(imputer_numeric, open(init_data_path + "/imputer_numeric.pkl", 'wb'))

    train = df
    for usecol in cat_cols:
        train[usecol] = train[usecol].astype('str')
        test[usecol] = test[usecol].astype('str')
        
        #Fit LabelEncoder
        le = LabelEncoder().fit(
                np.unique(train[usecol].unique().tolist()+ test[usecol].unique().tolist()))

        #At the end 0 will be used for dropped values
        train[usecol] = le.transform(train[usecol])+1
    
        train[usecol] = train[usecol].replace(np.nan, '').astype('int').astype('category')
       
    # SCALE
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df)
    #df = scaler.transform(df)
    pickle.dump(scaler, open(init_data_path + "/scaler.pkl", 'wb'))

    x = pd.DataFrame(df, columns=colonnes)
    varthresh = VarianceThreshold(threshold=x.var().describe()[4])  # 25% lowest variance threshold
    x = varthresh.fit_transform(x)
    pickle.dump(varthresh, open(init_data_path + "/varthresh.pkl", 'wb'))

    feature_after_vartresh = varthresh.get_feature_names_out(feature)
    final = pd.DataFrame(x, columns=feature_after_vartresh)

    final = drop_corolated_all(final)

    # drop not important
    #drop_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/exclude.csv")["feature"].values.tolist()
    #final = final.drop(drop_list, axis=1)

    # take top 150
    #top_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/top_150.csv")[
    #    "feature"].values.tolist()
    #final = final[top_list]

    final["pred"] = labels
    #final.to_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/ICU_after_preprocessing_impute_80.csv",
    #             index=False)
    final.to_csv(init_data_path+ "/after_preprocessing/ICU_after_preprocessing_impute_80.csv",index=False)
    
def preprocess_RADCOM(init_data_path):
    train = pd.read_csv(init_data_path + "/original/radcom_dataset_with_ood_labels.csv")

    if "Unnamed: 0" in train.columns:
        train = train.drop(["Unnamed: 0"], axis=1)

    # Extracting columns to change to Categorical
    col_train = train.columns
    
    spesial_cols = []
    spesial_cols.append(['protocol','res_label','serv_label','serv_ood_label','res_ood_label'])            
    
    train = train.drop(['serv_ood_label','res_ood_label','itag','code_name'], axis=1)
    train = train.drop(train[pd.isnull(train['index'])].index)

    #convert multy class to binary class
    train['res_label'] = np.where(train['res_label']<=2, 0, 1)
    
    cat_cols = list(['serv_label'])
    train[cat_cols] = train[cat_cols].apply(lambda x: x.astype('category'), axis=0)
    for usecol in cat_cols:
        train[usecol] = train[usecol].astype('str')
    
        #Fit LabelEncoder
        le = LabelEncoder().fit(
                np.unique(train[usecol].unique().tolist()))

        #At the end 0 will be used for dropped values
        train[usecol] = le.transform(train[usecol])+1
        train[usecol] = train[usecol].replace(np.nan, '').astype('int').astype('category')

    # datetime features
    train['start_of_peak'] = pd.to_datetime(train['start_of_peak'])
    # year is not importent
    train['start_of_peak_month'] = train['start_of_peak'].apply(lambda x: x.strftime('%m')) 
    train['start_of_peak_day'] = train['start_of_peak'].apply(lambda x: x.strftime('%d')) 
    train['start_of_peak_hour'] = train['start_of_peak'].apply(lambda x: x.strftime('%H'))
    train['start_of_peak_minute'] = train['start_of_peak'].apply(lambda x: x.strftime('%M')) 
    train['start_of_peak_second'] = train['start_of_peak'].apply(lambda x: x.strftime('%S'))  
    train['end_of_peak'] = pd.to_datetime(train['end_of_peak'])
    train['end_of_peak_month'] = train['end_of_peak'].apply(lambda x: x.strftime('%m')) 
    train['end_of_peak_day'] = train['end_of_peak'].apply(lambda x: x.strftime('%d')) 
    train['end_of_peak_hour'] = train['end_of_peak'].apply(lambda x: x.strftime('%H'))
    train['end_of_peak_minute'] = train['end_of_peak'].apply(lambda x: x.strftime('%M')) 
    train['end_of_peak_second'] = train['end_of_peak'].apply(lambda x: x.strftime('%S'))  
   
    # SCALE

    labels = train['res_label']
    df = train.drop(['res_label','start_of_peak','end_of_peak'], axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df)
    #df = scaler.transform(df)
    pickle.dump(scaler, open(init_data_path + "/scaler.pkl", 'wb'))

    col_train = train.columns
    feature = list(df.columns)
    x = pd.DataFrame(df, columns=col_train)
    varthresh = VarianceThreshold(threshold=x.var().describe()[4])  # 25% lowest variance threshold
    x = varthresh.fit_transform(x)
    pickle.dump(varthresh, open(init_data_path + "/varthresh.pkl", 'wb'))

    feature_after_vartresh = varthresh.get_feature_names_out(col_train)
    final = pd.DataFrame(x, columns=feature_after_vartresh)

    final = drop_corolated_all(final)

    # drop not important
    #drop_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/exclude.csv")["feature"].values.tolist()
    #final = final.drop(drop_list, axis=1)

    # take top 150
    #top_list = pd.read_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/top_150.csv")[
    #    "feature"].values.tolist()
    #final = final[top_list]

    final["pred"] = labels
    #final.to_csv("/".join(init_data_path.split("/")[:-1]) + "/after_preprocessing/ICU_after_preprocessing_impute_80.csv",
    #             index=False)
    final.to_csv(init_data_path+ "/after_preprocessing/RADCOM_after_preprocessing_no_year.csv",index=False)

'''   
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVC
from tqdm import tqdm as tqdm
import glob

def foo (init_data_path):

    script_id = 0

    data_path = init_data_path+'"/original'

    train = pd.read_csv(os.path.join(data_path, 'training_v2.csv'))

    id_col = 'encounter_id'

    fd = pd.read_csv(os.path.join(data_path, 'WiDS Datathon 2020 Dictionary.csv'))
    fd = fd[(fd['Data Type'] == 'string') | (fd['Data Type'] == 'binary')]
    cat_features = list(fd['Variable Name'].values)
    for c in cat_features:
        if c not in train.columns or c == 'hospital_death':
            cat_features.remove(c)
    print(f'cat_features: {cat_features} ({len(cat_features)})')


    extracted_files = glob.glob('./*.csv')
    extracted_files = [f[2:-8] for f in extracted_files]
    print(extracted_files)
    # error

    target_cols = []
    for c in train.columns:
        if c != id_col and c != 'hospital_death' and train[c].isnull().mean() > 0 and c not in extracted_files and c not in cat_features:
            target_cols.append({'fname': c, 'type': 'regression'})

    print(target_cols)

    def preprocess(df, min_max_cols):
        for c in min_max_cols:
            vals = df[[c, c.replace('_min', '_max')]].values.copy()
            df[c] = np.nanmin(vals, axis=1)
            df[c.replace('_min', '_max')] = np.nanmax(vals, axis=1)

    for t_i, target_data in enumerate(target_cols):
        target_col = target_data['fname']
        #dprint(f'********************************* {target_col} ({t_i+1}/{len(target_cols)}) *********************************')

        train = pd.read_csv(os.path.join(data_path, 'training_v2.csv'))
        test = pd.read_csv(os.path.join(data_path, 'unlabeled.csv'))

        min_max_cols = []
        for c in train.columns:
            if '_min' in c and c.replace('min', 'max') in train.columns:
                min_max_cols.append(c)
        print(f'min_max_cols: {min_max_cols} ({len(min_max_cols)})')

        preprocess(train, min_max_cols)
        preprocess(test, min_max_cols)

        print(f'Number of missing values in train: {train[target_col].isnull().mean()}')
        print(f'Number of missing values in test: {test[target_col].isnull().mean()}')

        train['is_test'] = 0
        test['is_test'] = 1
        df_all = pd.concat([train, test], axis=0)

        #dprint('Label Encoder...')
        cols = [f_ for f_ in df_all.columns if df_all[f_].dtype == 'object']
        print(cols)
        cnt = 0
        for c in tqdm(cols):
            if c != id_col and c != target_col:
                # print(c)
                le = LabelEncoder()
                df_all[c] = le.fit_transform(df_all[c].astype(str))
                cnt += 1

                del le
        #dprint('len(cols) = {}'.format(cnt))

        train = df_all.loc[df_all['is_test'] == 0].drop(['is_test'], axis=1)
        test = df_all.loc[df_all['is_test'] == 1].drop(['is_test'], axis=1)

        # del df_all
        # gc.collect()

        # Rearrange train and test
        train = df_all[np.logical_not(df_all[target_col].isnull())].drop(['is_test'], axis=1)
        test = df_all[df_all[target_col].isnull()].drop(['is_test'], axis=1)
        #dprint(train.shape, test.shape) 

        if target_data['type'] == 'classification':
            tle = LabelEncoder()
            train[target_col] = tle.fit_transform(train[target_col].astype(str))

        empty_cols = []
        for c in test.columns:
            n = (~test[c].isnull()).sum()
            if n == 0:
                empty_cols.append(c)
        print(f'empty_cols: {empty_cols}')


        # error
        features = list(train.columns.values)
        features.remove(id_col)
        features.remove(target_col)


        # Build the model
        cnt = 0
        p_buf = []
        n_splits = 4
        n_repeats = 1
        kf1 = RepeatedKFold(
            n_splits=n_splits, 
            n_repeats=n_repeats, 
            random_state=0)
        kf2 = RepeatedKFold(
            n_splits=n_splits, 
            n_repeats=n_repeats, 
            random_state=1)
        err_buf = []   
        undersampling = 0

        if target_data['type'] == 'regression':
            lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mse',
                'max_depth': 8,
                'learning_rate': 0.05, 
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85,
                'bagging_freq': 5,
                'lambda_l1': 1.0,
                'lambda_l2': 1.0,
                'verbose': -1,
                'num_threads': -1,
            }
        elif target_data['type'] == 'classification':
            #dprint(f'Num classes: {train[target_col].nunique()} ({train[target_col].unique()})')
            if train[target_col].nunique() == 2:
                lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'max_depth': 8,
                    'learning_rate': 0.05, 
                    'feature_fraction': 0.85,
                    'bagging_fraction': 0.85,
                    'bagging_freq': 5,
                    'lambda_l1': 1.0,
                    'lambda_l2': 1.0,
                    'verbose': -1,
                    'num_threads': -1,
                }
            else:
                lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'max_depth': 8,
                    'learning_rate': 0.05, 
                    'feature_fraction': 0.85,
                    'bagging_fraction': 0.85,
                    'bagging_freq': 5,
                    'lambda_l1': 1.0,
                    'lambda_l2': 1.0,
                    'verbose': -1,
                    'num_threads': -1,
                    'num_class': train[target_col].nunique()
                }

        cols_to_drop = [
            id_col,
            target_col,
            'hospital_death',
            # 'bmi',
        ] + empty_cols

        # cols_to_use = features
        X = train.drop(cols_to_drop, axis=1, errors='ignore')
        y = train[target_col].values
        id_train = train[id_col].values

        X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
        id_test = test[id_col].values

        feature_names = list(X.columns)
        n_features = X.shape[1]
        dprint(f'n_features: {n_features}')

        p_test = []
        dfs_train = []
        dfs_test = []

        for fold_i_oof, (train_index_oof, valid_index_oof) in enumerate(kf1.split(X, y)):
            x_train_oof = X.iloc[train_index_oof]
            x_valid_oof = X.iloc[valid_index_oof]

            y_train_oof = y[train_index_oof]
            y_valid_oof = y[valid_index_oof]

            id_train_oof = id_train[valid_index_oof]

            for fold_i, (train_index, valid_index) in enumerate(kf2.split(x_train_oof, y_train_oof)):
                params = lgb_params.copy() 

                x_train = x_train_oof.iloc[train_index]
                x_valid = x_train_oof.iloc[valid_index]

                lgb_train = lgb.Dataset(
                    x_train, 
                    y_train_oof[train_index], 
                    feature_name=feature_names,
                    )
                lgb_train.raw_data = None

                lgb_valid = lgb.Dataset(
                    x_valid, 
                    y_train_oof[valid_index],
                    )
                lgb_valid.raw_data = None

                model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=[lgb_valid],
                    early_stopping_rounds=100,
                    verbose_eval=100,
                )

                if fold_i_oof == 0:
                    importance = model.feature_importance()
                    model_fnames = model.feature_name()
                    tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
                    tuples = [x for x in tuples if x[1] > 0]
                    print('Important features:')
                    for i in range(20):
                        if i < len(tuples):
                            print(tuples[i])
                        else:
                            break

                    del importance, model_fnames, tuples

                p_lgbm = model.predict(x_valid, num_iteration=model.best_iteration)
                if target_data['type'] == 'regression':
                    err = mean_squared_error(y_train_oof[valid_index], p_lgbm)
                    err_buf.append(err)
                    dprint('{} LGBM MSE: {:.4f}'.format(fold_i, err))
                elif target_data['type'] == 'classification':
                    if train[target_col].nunique() == 2:
                        err = roc_auc_score(y_train_oof[valid_index], p_lgbm)
                        dprint('{} LGBM AUC: {:.6f}'.format(fold_i, err))
                    err = log_loss(y_train_oof[valid_index], p_lgbm)
                    err_buf.append(err)
                    dprint('{} LGBM LOSS: {:.4f}'.format(fold_i, err))

                p_lgbm_train = model.predict(x_valid_oof, num_iteration=model.best_iteration)
                p_lgbm_test = model.predict(X_test[feature_names], num_iteration=model.best_iteration)

                df_train = pd.DataFrame()
                df_train[id_col] = id_train_oof
                if target_data['type'] == 'regression':
                    df_train[target_col] = p_lgbm_train
                elif target_data['type'] == 'classification':
                    if train[target_col].nunique() == 2:
                        df_train[target_col] = p_lgbm_train
                    else:
                        for i, t in enumerate(np.sort(train[target_col].unique())):
                            df_train[str(t)] = p_lgbm_train[:, i]

                dfs_train.append(df_train)

                df_test = pd.DataFrame()
                df_test[id_col] = id_test
                if target_data['type'] == 'regression':
                    df_test[target_col] = p_lgbm_test
                elif target_data['type'] == 'classification':
                    if train[target_col].nunique() == 2:
                        df_test[target_col] = p_lgbm_test
                    else:
                        for i, t in enumerate(np.sort(train[target_col].unique())):
                            df_test[str(t)] = p_lgbm_test[:, i]

                dfs_test.append(df_test)
                
                # p_test.append(p_lgbm_test)

                del model, lgb_train, lgb_valid
                gc.collect

            # break

        err_mean = np.mean(err_buf)
        err_std = np.std(err_buf)
        dprint('ERR: {:.4f} +/- {:.4f}'.format(err_mean, err_std))

        dfs_train = pd.concat(dfs_train, axis=0)
        if target_data['type'] == 'regression':
            dfs_train = dfs_train.groupby(id_col)[target_col].mean().reset_index().rename({target_col: target_col + '_est'}, axis=1)
        elif target_data['type'] == 'classification':
            if train[target_col].nunique() == 2:
                dfs_train = dfs_train.groupby(id_col)[target_col].mean().reset_index()
                dfs_train[target_col] = tle.inverse_transform(np.round(dfs_train[target_col].values).astype(int))
                dfs_train.rename({target_col: target_col + '_est'}, inplace=True, axis=1)
            else:
                dfs_train = dfs_train.groupby(id_col).mean().reset_index()
                cols = np.sort(train[target_col].unique()).astype(str)
                dfs_train[target_col + '_est'] = tle.inverse_transform(np.argmax(dfs_train[cols].values, axis=1))
        print(dfs_train.head())

        dfs_test = pd.concat(dfs_test, axis=0)
        if target_data['type'] == 'regression':
            dfs_test = dfs_test.groupby(id_col)[target_col].mean().reset_index().rename({target_col: target_col + '_est'}, axis=1)
        elif target_data['type'] == 'classification':
            if train[target_col].nunique() == 2:
                dfs_test = dfs_test.groupby(id_col)[target_col].mean().reset_index()
                dfs_test[target_col] = tle.inverse_transform(np.round(dfs_test[target_col].values).astype(int))
                dfs_test.rename({target_col: target_col + '_est'}, inplace=True, axis=1)
            else:
                dfs_test = dfs_test.groupby(id_col).mean().reset_index()
                cols = np.sort(train[target_col].unique()).astype(str)
                dfs_test[target_col + '_est'] = tle.inverse_transform(np.argmax(dfs_test[cols].values, axis=1))
        print(dfs_test.head())

        out = pd.concat([dfs_train, dfs_test], axis=0)
        out.to_csv(target_col + '_est.csv', index=False)
        '''