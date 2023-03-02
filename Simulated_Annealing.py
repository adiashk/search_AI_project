"""
Simulated Annealing Class
"""
import pickle
import random
import math

import numpy as np
import sklearn
import pandas as pd

import configparser
import random
from pathlib import Path
import joblib

from Utils.attack_utils import get_constrains
from Models.scikitlearn_wrapper import SklearnClassifier
from Utils.data_utils import split_to_datasets

def get_config():
    config = configparser.ConfigParser()
    # config.read(sys.argv[1])
    config.read('configurations.txt')
    config = config['DEFAULT']
    return config


def date_change(current):
    # year and month are not change. only the day
    dates = []
    new_date = current.copy()  # 20180200
    while (
            new_date / 100 == current / 100 and new_date % 100 <= 30):  # stay in same year and month, day can increase until 30
        new_date = new_date + 1
        dates.append(new_date)

    return dates


def time_change(current):
    new_time = current.copy()
    times = []
    new_date = current.copy()  # 235959

    while (new_time / 10000 < 24):
        while ((new_time / 100) % 100 < 60):
            while (new_time % 100 < 60):
                new_time = new_time + 29  # should be 1
                times.append(new_time)
            new_time = (new_time / 100 + 2) * 100  # add minute #should be +1
            times.append(new_time)
        new_time = (new_time / 10000 + 1) * 10000  # add hour
        times.append(new_time)

    return times


def get_feature_range(dataset_name):
    feature_range = {}
    if dataset_name == "RADCOM":
        # feature_range = {
        #     'agg_count': range(1, 300, 1),  # 0
        #     'delta_delta_delta_from_previous_request': range(0, 1000, 10),  # 100000, 1 # 1
        #     'delta_delta_from_previous_request': range(0, 1000, 10),  # 2
        #     'delta_from_previous_request': range(0, 1000, 10),  # 3
        #     'delta_from_start': range(0, 1000, 10),  # 4
        #     'effective_peak_duration': range(0, 1000, 10),  # 100000, 0.01  # 5
        #     # 'index':range(), # 6
        #     # 'minimal_bit_rate':range(), # 7
        #     'non_request_data': range(0, 100, 1), # 8
        #     # 'peak_duration':range(), # 9
        #     # 'peak_duration_sum':range(), # 10
        #     'previous_previous_previous_previous_total_sum_of_data_to_sec': range(0, 100000, 1000),  # 100000000 # 11
        #     'previous_previous_previous_total_sum_of_data_to_sec': range(0, 100000, 1000),  # 12
        #     'previous_previous_total_sum_of_data_to_sec': range(0, 100000, 1000),  # 13
        #     'previous_total_sum_of_data_to_sec': range(0, 100000, 1000),  # 14
        #     'sum_of_data': range(0, 100000, 1000),  # 100000000, 1 # 15
        #     'total_sum_of_data': range(0, 100000, 1000), # 100000000, 1 # 16
        #     'total_sum_of_data_to_sec': range(0, 1000, 10),  # 1000000, 1 # 17
        #     'serv_label': range(1, 3, 1),  # 0,1,2 # 18
        #     'start_of_peak_date': date_change(),  # 19
        #     'start_of_peak_time': date_change(),  # 20
        #     'end_of_peak_date': time_change(),  # 21
        #     'end_of_peak_time': time_change(),  # 22
        # }
        feature_range = {
            'previous_previous_previous_previous_total_sum_of_data_to_sec': range(0, 100000, 100),  # 100000000 # 11
            'previous_previous_previous_total_sum_of_data_to_sec': range(0, 100000, 100),  # 12
            'previous_previous_total_sum_of_data_to_sec': range(0, 100000, 100),  # 13
            'previous_total_sum_of_data_to_sec': range(0, 100000, 100),  # 14
            'total_sum_of_data_to_sec': range(0, 1000, 10),  # 1000000, 1 # 17

        }
    elif dataset_name == "HATE":
        feature_range = {
            
            'c_work_empath': np.linspace(0.1, 0.9, 100),
            'normal_neigh': np.linspace(0.1, 0.9, 100),
            'c_legend_empath': np.linspace(0.1, 0.9, 100),
            'c_cleaning_empath': np.linspace(0.1, 0.9, 100),
            'sleep_empath': np.linspace(0.1, 0.9, 100),
            'c_furniture_empath': np.linspace(0.1, 0.9, 100),
            'c_ridicule_empath': np.linspace(0.1, 0.9, 100),
            'c_fire_empath': np.linspace(0.1, 0.9, 100),
            'hate_neigh': np.linspace(0.1, 0.9, 100),
        }
        """
            'sports_empath': np.linspace(0.1, 0.9, 100),
            'statuses_count': np.linspace(0, 1000, 10),
            'surprise_empath': np.linspace(0.1, 0.9, 100),
            'tourism_empath': np.linspace(0.1, 0.9, 100),
            'urban_empath': np.linspace(0.1, 0.9, 100),
            'vacation_empath': np.linspace(0.1, 0.9, 100),
            'warmth_empath': np.linspace(0.1, 0.9, 100),
            'work_empath': np.linspace(0.1, 0.9, 100),
            'youth_empath': np.linspace(0.1, 0.9, 100),
            'zest_empath': np.linspace(0.1, 0.9, 100),
        """

            

    elif dataset_name == 'CREDIT':
        feature_range = {
            'PREV_ACTIVE_INSTALMENT_PAYMENT_DIFF_MEAN': np.linspace(0.1, 0.9, 100),
            'PREV_Consumer_AMT_CREDIT_SUM': np.linspace(0.1, 0.9, 100),
            #'PREV_NAME_CONTRACT_STATUS_Refused_MEAN': np.linspace(0.1, 0.9, 10),
            'NAME_EDUCATION_TYPE': {0,0.25,0.5,0.75},
            'AMT_ANNUITY': np.linspace(0.1, 0.9, 100),
            'PREV_Cash_SIMPLE_INTERESTS_MEAN': np.linspace(0.1, 0.9, 100),
            'CREDIT_TO_GOODS_RATIO': np.linspace(0.1, 0.9, 100),
            'DAYS_EMPLOYED': np.linspace(0.1, 0.9, 100),
            'CREDIT_TO_ANNUITY_RATIO': np.linspace(0.1, 0.9, 100),
        }
    return feature_range





class SimulatedAnnealing:
    def __init__(self, initialSolution, solutionEvaluator, initialTemp, finalTemp, tempReduction, neighborOperator=None,
                 iterationPerTemp=200, alpha=10, beta=5, record_id=0, record_true_class=0, model_name=""):
        self.solution = initialSolution
        self.evaluate = solutionEvaluator
        self.initialTemp = initialTemp
        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        self.iterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.beta = beta
        self.neighborOperator = self.neighbor_operator_func
        self.record_id = record_id
        self.record_true_class = record_true_class
        df_temp = pd.DataFrame(self.solution).T
        self.path_to_file = "results/" + model_name + f"/solution_{self.record_id}_{self.record_true_class}.csv"
        output_dir = Path("results/" + model_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_temp.to_csv(self.path_to_file, index=False)
        self.max_cost = self.evaluate(self.solution.values.reshape(1, -1))[0][self.record_true_class]
        self.best_solution = self.solution
        

        if tempReduction == "linear":
            self.decrementRule = self.linearTempReduction
        elif tempReduction == "geometric":
            self.decrementRule = self.geometricTempReduction
        elif tempReduction == "slowDecrease":
            self.decrementRule = self.slowDecreaseTempReduction
        else:
            self.decrementRule = tempReduction


    def linearTempReduction(self):
        self.currTemp -= self.alpha

    def geometricTempReduction(self):
        self.currTemp *= self.alpha

    def slowDecreaseTempReduction(self):
        self.currTemp = self.currTemp / (1 + self.beta * self.currTemp)

    def isTerminationCriteriaMet(self):
        # can add more termination criteria
        return self.currTemp <= self.finalTemp or self.neighborOperator(self.solution) == 0

    def neighbor_operator_func(self, current):
        # return all neighbor of cuurent
        # neighbor is a sample that differ from current in one editable feature
        editable = perturbability
        neighbors = []
        for feature in editable.Row:  # for each feature
            if editable[editable['Row'] == feature]['Perturbability'].values[0] == 1:  # the feature can be edited
                if feature in feature_range:
                    for change in feature_range[feature]:
                        neighbor = current.copy()
                        if neighbor[feature] != change:  # different value for specific feature
                            neighbor[feature] = change
                            neighbors.append(neighbor)

        return neighbors
    
    def run(self):
        while not self.isTerminationCriteriaMet():
            new_sol_value = 0
            # iterate that number of times, based on the temperature
            for i in range(self.iterationPerTemp):
                # get all the neighbors
                neighbors = self.neighborOperator(self.solution)
                if len(neighbors) == 0:
                    continue
                # print("Number of neighbors: ", len(neighbors))
                '''
                # pick a random neighbor
                # newSolution = random.choice(neighbors)
                '''
                # get 10 random neighbors and pick the best one -> minimal cost
                reandom_neighbors = random.sample(neighbors, 100)
                # predict the cost of each neighbor and get the solution with the minimal cost -> the best neighbor
                # neighbors_cost = []
                # for neighbor in reandom_neighbors:
                #     neighbors_cost.append(self.evaluate(neighbor.values.reshape(1, -1))[0][self.record_true_class])
                # newSolution = reandom_neighbors[np.argmin(neighbors_cost)]

                newSolution = reandom_neighbors[np.argmin(self.evaluate(reandom_neighbors), axis=0)[self.record_true_class]]

                # df_temp = pd.DataFrame(newSolution).T
                # df_old_sols = pd.read_csv(self.path_to_file)
                # all_df = pd.concat([df_old_sols, df_temp], axis=0, ignore_index=True)
                '''
                #  check if the neighbor is already in the path
                old_shape = all_df.shape
                all_df.drop_duplicates(inplace=True)
                if old_shape != all_df.shape:  # duplicate -> new neighbor in path already -> do not add to neighbors
                    continue
                # no duplicate -> new neighbor not in path:
                '''
                # get the cost between the two solutions
                # cost = self.evaluate(self.solution) - self.evaluate(newSolution)
                curr_sol_val = self.evaluate(self.solution.values.reshape(1, -1))[0][self.record_true_class]
                new_sol_val = self.evaluate(newSolution.values.reshape(1, -1))[0][self.record_true_class]
                if new_sol_val < 0.5:
                    print("find attacked sample!!!")
                    #print("Best Cost: ", new_sol_val)
                    
                    return [1, newSolution]
                cost = curr_sol_val - new_sol_val
                # if the new solution is better, accept it
                if cost >= 0:
                    self.solution = newSolution
                    # self.path = pd.concat([self.path, self.solution], axis=1)
                    # self.path_score.append(new_sol_val)
                    # all_df.to_csv(self.path_to_file, index=False)
                    if new_sol_val < self.max_cost:  # new best solution
                        self.max_cost = new_sol_val
                        self.best_solution = self.solution
                        # self.currTemp = self.initialTemp
                        print("Best Cost: ", self.evaluate(self.solution.values.reshape(1, -1))[0][self.record_true_class])



                # if the new solution is not better, accept it with a probability of e^(-cost/temp)
                else:
                    if random.uniform(0, 0.7) < math.exp(-cost / (self.currTemp*2)):
                        self.solution = newSolution
                        # self.path = pd.concat([self.path, self.solution], axis=1)
                        # self.path_score.append(new_sol_val)
                        #all_df.to_csv(self.path_to_file, index=False)
            #print("Current Temperature: ", self.currTemp)
            print("Current Cost: ", self.evaluate(self.solution.values.reshape(1, -1))[0][self.record_true_class])
            '''
            if new_sol_val > self.max_cost:  # current solution is not the best
                self.currTemp += self.alpha  # increase temperature because we are not improving
                self.solution = self.best_solution
            '''
            # decrement the temperature
            self.decrementRule()
        if self.neighborOperator(self.solution) == 0:
            print('no neighbors')
        return[0, None]
        




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

    datasets = split_to_datasets(raw_data_path, save_path=data_path)
    x_attack = datasets.get("x_test")
    y_attack = datasets.get("y_test")

    if ('RADCOM' in dataset_name):
        x_attack = pd.read_csv('Datasets/RADCOM/x_test_seed_42_val_size_0.25_surrgate_train_size_0.5.csv')
        y_attack = pd.read_csv('Datasets/RADCOM/y_test_seed_42_val_size_0.25_surrgate_train_size_0.5.csv')
        # model = pickle.load(open('Models/RADCOM/RADCOM_target_GB_seed-42_lr-0.01_estimators-500_maxdepth-9.pkl', 'rb'))
        model = pickle.load(open('Models/RADCOM/RADCOM_target_RF_seed-42_estimators-500_maxdepth-9.pkl', 'rb'))
        # model = pickle.load(open('RADCOM_target_XGB_seed-42_lr-0.1_estimators-70_maxdepth-8', 'rb'))
    elif ('HATE' in dataset_name):
        x_attack = pd.read_csv('Datasets/HATE/x_test_seed_42_val_size_0.25_surrgate_train_size_0.5.csv')
        y_attack = pd.read_csv('Datasets/HATE/y_test_seed_42_val_size_0.25_surrgate_train_size_0.5.csv')
        #x_attack = pd.read_csv('Datasets/HATE/x_orig_attack.csv')
        #y_attack = pd.read_csv('Datasets/HATE/y_orig_attack.csv')
        #model = joblib.load('Models/HATE/rf_sota_model.pkl')
        #model = pickle.load(open('Models/HATE/HATE_target_RF_seed-42_estimators-100_maxdepth-3.pkl', 'rb'))
        #model = pickle.load(open('Models/HATE/HATE_target_XGB_seed-42_lr-0.1_estimators-70_maxdepth-8.pkl', 'rb'))
        model = pickle.load(open('Models/HATE/HATE_target_GB_seed-42_lr-1.0_estimators-100_maxdepth-3.pkl', 'rb'))
    elif ('CREDIT' in dataset_name):
        x_attack = pd.read_csv('Datasets/CREDIT/x_test_seed_42_val_size_0.25_surrgate_train_size_0.5.csv')
        y_attack = pd.read_csv('Datasets/CREDIT/y_test_seed_42_val_size_0.25_surrgate_train_size_0.5.csv')
        #x_attack = pd.read_csv('Datasets/HATE/x_orig_attack.csv')
        #y_attack = pd.read_csv('Datasets/HATE/y_orig_attack.csv')
        model = pickle.load(open('Models/CREDIT/CREDIT_target_RF_seed-42_estimators-200_maxdepth-9.pkl', 'rb'))
        #model = pickle.load(open('Models/CREDIT/CREDIT_target_GB_seed-42_lr-1.0_estimators-100_maxdepth-3.pkl', 'rb'))

    perturbability = pd.read_csv(perturbability_path)
    feature_range = get_feature_range(dataset_name)

    model_name = model.__class__.__name__
    print("model name: ", model_name)
    
    attack_x = datasets.get("x_test")
    attack_y = datasets.get("y_test")
    preds = model.predict(attack_x)
    eq = np.equal(preds,attack_y['pred'])
    i=0

    #target_model = SklearnClassifier(model=target, columns=attack_x.columns)
    #constrains, perturbability = get_constrains(dataset_name, perturbability_path)
    columns_names = list(attack_x.columns)
    #random.seed(seed)
    #np.random.seed(seed)
    #print (target_models_names[j])
    num_success = 0
    mis = 0
    well = 0
    attack_set = []
    while i < attack_x.shape[0]:  # 10 random records to attack

        record = attack_x.loc[i]
        record_true_class = int(attack_y.pred[i])
        record_pred = int(preds[i])
        #print("true label: ", int(record_true_class))
        prediction_record = model.predict(record.values.reshape(1, -1))[0]
        if (record_pred != prediction_record):
            print('pred != pred')
        if (record_pred != record_true_class):
            print("record is misclassified")
            mis += 1
            i += 1
            continue
        
        i += 1
        well +=1
        
        SA = SimulatedAnnealing(initialSolution=record, solutionEvaluator=model.predict_proba,
                                initialTemp=100, finalTemp=0.01,
                                tempReduction="linear",
                                iterationPerTemp=100, alpha=10, beta=5, record_id=i, record_true_class=int(record_true_class), 
                                model_name=model_name)
        attack_res = SA.run()
        if (attack_res[0] == 1):
            num_success = num_success+1
            rec = list((attack_res[1].values.reshape(1,-1)).flatten())
            attack_set.append(rec)
        
        #print("final solution for sample : ", SA.max_cost)
        print("======================+=======================")
    print(i, " samples")
    print(num_success, " samples success attack")
    print(mis, "mis")
    print(well, "well")
    attack_sets = pd.DataFrame(attack_set, columns=columns_names)
    attack_sets.to_csv("Datasets/HATE/HATE_adv_"+ model_name +".csv",index=False)
    