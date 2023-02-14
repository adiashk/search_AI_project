import sys
import pandas as pd
import numpy as np
import configparser
import random

import sklearn

#from Models.scikitlearn_wrapper import SklearnClassifier
from Utils.anomaly_utils import get_ocsvm
#from Utils.attack_utils import get_hopskipjump, get_constrains
from Utils.data_utils import split_to_datasets, preprocess_ICU, preprocess_RADCOM
from Utils.models_utils import train_GB_model, train_RF_model
from sklearn.svm import OneClassSVM



def get_config():
    config = configparser.ConfigParser()
    #config.read(sys.argv[1])
    config.read('configurations.txt')
    config = config['DEFAULT']
    return config


def main_run_attack():
    # Set parameters
    configurations = get_config()
    data_path = configurations["data_path"]
    raw_data_path = configurations["raw_data_path"]
    perturbability_path = configurations["perturbability_path"]
    results_path = configurations["results_path"]
    seed = int(configurations["seed"])
    exclude = configurations["exclude"]
    dataset_name = raw_data_path.split("/")[1]

    # Train models
    if exclude == "None":
        datasets = split_to_datasets(raw_data_path, save_path=data_path)
        GB, gb_eval = train_GB_model(data_path, datasets=datasets, model_type="target")
        RF, rf_eval = train_RF_model(data_path, datasets=datasets, model_type="target")
    else:
        datasets = split_to_datasets(raw_data_path, save_path=data_path, exclude=exclude)
        GB, gb_eval = train_GB_model(data_path, datasets=datasets, model_type="target", exclude=exclude)
        RF, rf_eval = train_RF_model(data_path, datasets=datasets, model_type="target", exclude=exclude)

    # target_models = [RF]
    # target_models_names = ["RF"]
    target_models = [GB, RF]
    target_models_names = ["GB", "RF"]

    for j, target in enumerate(target_models):
        # Attack preparation
        attack_x = datasets.get("x_test")
        target_model = SklearnClassifier(model=target, columns=attack_x.columns)
        constrains, perturbability = get_constrains(dataset_name, perturbability_path)
        columns_names = list(attack_x.columns)
        random.seed(seed)
        np.random.seed(seed)

        # Attack
        attack = get_hopskipjump(target_model, constrains, columns_names)
        adv = attack.generate(x=attack_x.to_numpy(), perturbability=perturbability)
        original = attack_x.to_numpy()
        true_label = datasets.get("y_test").transpose().values.tolist()[0]
        pred_original = target_model.predict(original)
        pred_adv = target_model.predict(adv)

        # Save Results
        results_dict = {
            "original_pred": target_model.predict(original).argmax(1),
            "true_label": true_label,
            "adv_pred": target_model.predict(adv).argmax(1),
        }
        for i, col in enumerate(list(attack_x.columns)):
            results_dict.update({col: adv[:, i]})

        pd.DataFrame(results_dict).to_csv(
            results_path + "/hopskipjump_{}_exclude_{}.csv".format(target_models_names[j], exclude), index=False)

import parse
if __name__ == '__main__':

    # Set parameters
    configurations = get_config()
    data_path = configurations["data_path"]
    raw_data_path = configurations["raw_data_path"]
    perturbability_path = configurations["perturbability_path"]
    results_path = configurations["results_path"]
    seed = int(configurations["seed"])
    exclude = configurations["exclude"]
    dataset_name = raw_data_path.split("/")[1]

    #preprocess_RADCOM(raw_data_path)
    
    # Train models
    if exclude == "None":
        datasets = split_to_datasets(raw_data_path, save_path=data_path)
        GB, gb_eval = train_GB_model(data_path, datasets=datasets, model_type="target")
        RF, rf_eval = train_RF_model(data_path, datasets=datasets, model_type="target")
    else:
        datasets = split_to_datasets(raw_data_path, save_path=data_path, exclude=exclude)
        GB, gb_eval = train_GB_model(data_path, datasets=datasets, model_type="target", exclude=exclude)
        RF, rf_eval = train_RF_model(data_path, datasets=datasets, model_type="target", exclude=exclude)


    # target_models = [RF]
    # target_models_names = ["RF"]
    target_models = [GB, RF]
    target_models_names = ["GB", "RF"]

    for j, target in enumerate(target_models):
        # Attack preparation
        attack_x = datasets.get("x_test")
        target_model = SklearnClassifier(model=target, columns=attack_x.columns)
        constrains, perturbability = get_constrains(dataset_name, perturbability_path)
        columns_names = list(attack_x.columns)
        random.seed(seed)
        np.random.seed(seed)

        # Attack
        attack = get_hopskipjump(target_model, constrains, columns_names)
        adv = attack.generate(x=attack_x.to_numpy(dtype=np.float64), perturbability=perturbability)
        original = attack_x.to_numpy()
        true_label = datasets.get("y_test").transpose().values.tolist()[0]
        pred_original = target_model.predict(original)
        pred_adv = target_model.predict(adv)

        # Save Results
        results_dict = {
            "original_pred": target_model.predict(original).argmax(1),
            "true_label": true_label,
            "adv_pred": target_model.predict(adv).argmax(1),
        }
        for i, col in enumerate(list(attack_x.columns)):
            results_dict.update({col: adv[:, i]})

        pd.DataFrame(results_dict).to_csv(
            results_path + "/hopskipjump_{}_exclude_{}.csv".format(target_models_names[j], exclude), index=False)

    #     ______Anomaly______
    # for j, target in enumerate(target_models):
    #     train_x = datasets["x_train_target"]
    #     attacked_data = pd.read_csv("Results/HATE/hopskipjump_GB_exclude_None.csv")
    #     attacked_data_good = attacked_data[attacked_data["original_pred"] == attacked_data["true_label"]]
    #     attacked_data_good = attacked_data_good[attacked_data_good["original_pred"] != attacked_data_good["adv_pred"]]
    #     # ocsvm = OneClassSVM(nu=0.25).fit(train_x)
    #     # train_x["prediction"] = datasets["y_train_target"]["pred"].values
    #
    #     ocsvm = sklearn.ensemble.IsolationForest(random_state=seed).fit(train_x)
    #
    #     origin = pd.DataFrame({"t":ocsvm.predict(train_x)})
    #     attacked_data = attacked_data_good.drop(["original_pred","adv_pred","true_label"], axis=1)
    #     # attacked_data["prediction"] = GB.predict(attacked_data)
    #     pred = pd.DataFrame({"t":ocsvm.predict(attacked_data)})
    #     benign = datasets["x_test"].reset_index(drop=True).loc[attacked_data_good.index]
    #     # benign["prediction"] = GB.predict(benign)
    #     pred_benign = pd.DataFrame({"t":ocsvm.predict(benign)})
    #     print()


