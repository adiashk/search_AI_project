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

from Utils.attack_utils import get_constrains


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
        #     'agg_count': range(1, 300, 1),
        #     'delta_delta_delta_from_previous_request': range(0, 1000, 10),  # 100000, 1
        #     'delta_delta_from_previous_request': range(0, 1000, 10),
        #     'delta_from_previous_request': range(0, 1000, 10),
        #     'delta_from_start': range(0, 1000, 10),
        #     'effective_peak_duration': range(0, 1000, 10),  # 100000, 0.01
        #     # 'index':range(),
        #     # 'minimal_bit_rate':range(),
        #     'non_request_data': range(0, 100, 1),
        #     # 'peak_duration':range(),
        #     # 'peak_duration_sum':range(),
        #     'previous_previous_previous_previous_total_sum_of_data_to_sec': range(0, 100000, 1000),  # 100000000
        #     'previous_previous_previous_total_sum_of_data_to_sec': range(0, 100000, 1000),
        #     'previous_previous_total_sum_of_data_to_sec': range(0, 100000, 1000),
        #     'previous_total_sum_of_data_to_sec': range(0, 100000, 1000),
        #     'sum_of_data': range(0, 100000, 1000),  # 100000000, 1
        #     'total_sum_of_data': range(0, 100000, 1000),
        #     'total_sum_of_data_to_sec': range(0, 1000, 10),  # 1000000, 1
        #     'serv_label': range(1, 3, 1),  # 0,1,2
        #     'start_of_peak_date': date_change(),
        #     'start_of_peak_time': date_change(),
        #     'end_of_peak_date': time_change(),
        #     'end_of_peak_time': time_change(),
        # }
        feature_range = {
            'agg_count': range(1, 300, 1),
            'sum_of_data': range(0, 100000, 10),  # 100000000, 1
        }

    return feature_range





class SimulatedAnnealing:
    def __init__(self, initialSolution, solutionEvaluator, initialTemp, finalTemp, tempReduction, neighborOperator=None,
                 iterationPerTemp=200, alpha=10, beta=5):
        self.solution = initialSolution
        self.evaluate = solutionEvaluator
        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        self.iterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.beta = beta
        self.neighborOperator = self.neighbor_operator_func
        # self.solution.to_csv("solution.csv")
        df_temp = pd.DataFrame(self.solution).T
        df_temp.to_csv("solution.csv", index=False)
        # self.path = pd.DataFrame()
        # self.path = pd.concat([self.path, self.solution], axis=1)
        # self.path_score = []
        # self.path_score.append(self.evaluate(self.solution.values.reshape(1, -1))[0][int(record_pred)])

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
            # iterate that number of times, based on the temperature
            for i in range(self.iterationPerTemp):
                # get all the neighbors
                neighbors = self.neighborOperator(self.solution)
                if len(neighbors) == 0:
                    continue
                # print("Number of neighbors: ", len(neighbors))
                # pick a random neighbor
                newSolution = random.choice(neighbors)
                #  check if the neighbor is already in the path
                df_temp = pd.DataFrame(newSolution).T
                df_old_sols = pd.read_csv("solution.csv")
                all_df = pd.concat([df_old_sols, df_temp], axis=0, ignore_index=True)
                old_shape = all_df.shape
                all_df.drop_duplicates(inplace=True)
                if old_shape != all_df.shape:  # duplicate -> new neighbor in path already -> do not add to neighbors
                    continue
                # no duplicate -> new neighbor not in path:

                # get the cost between the two solutions
                # cost = self.evaluate(self.solution) - self.evaluate(newSolution)
                curr_sol_val = self.evaluate(self.solution.values.reshape(1, -1))[0][int(record_pred)]
                new_sol_val = self.evaluate(newSolution.values.reshape(1, -1))[0][int(record_pred)]
                cost = curr_sol_val - new_sol_val
                # if the new solution is better, accept it
                if cost >= 0:
                    self.solution = newSolution
                    # self.path = pd.concat([self.path, self.solution], axis=1)
                    # self.path_score.append(new_sol_val)
                    all_df.to_csv("solution.csv", index=False)

                # if the new solution is not better, accept it with a probability of e^(-cost/temp)
                else:
                    if random.uniform(0, 1) < math.exp(-cost / self.currTemp):
                        self.solution = newSolution
                        # self.path = pd.concat([self.path, self.solution], axis=1)
                        # self.path_score.append(new_sol_val)
                        all_df.to_csv("solution.csv", index=False)
            print("Current Temperature: ", self.currTemp)
            print("Current Cost: ", self.evaluate(self.solution.values.reshape(1, -1))[0][int(record_pred)])
            # decrement the temperature
            self.decrementRule()



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

    x_attack = pd.read_csv('Datasets/RADCOM/x_test_seed_42_val_size_0.25_surrgate_train_size_0.5.csv')
    y_attack = pd.read_csv('Datasets/RADCOM/y_test_seed_42_val_size_0.25_surrgate_train_size_0.5.csv')
    record_id = 13740
    record = x_attack.iloc[record_id]
    record_pred = y_attack.iloc[record_id]
    # record = x_attack.iloc[:,:]
    # record_pred = y_attack.iloc[:,:]
    print("true label: ", int(record_pred))

    # constrains, perturbability = get_constrains(dataset_name, perturbability_path)
    perturbability = pd.read_csv(perturbability_path)

    feature_range = get_feature_range(dataset_name)

    # model = pickle.load(open('Models/RADCOM/RADCOM_target_RF_seed-42_estimators-500_maxdepth-9.pkl', 'rb'))
    model = pickle.load(open('Models/RADCOM/RADCOM_target_GB_seed-42_lr-0.01_estimators-500_maxdepth-9.pkl', 'rb'))

    print("prediction: ", int(model.predict(record.values.reshape(1, -1))[0]))
    print("prediction prob: ", model.predict_proba(record.values.reshape(1, -1))[0])
    # print(model.predict_proba(record.values.reshape(1, -1))[0][int(record_pred)])
    # path_cost = model.predict_proba(record.values.reshape(1, -1))[0][int(record_pred)]  # as func

    SA = SimulatedAnnealing(initialSolution=record, solutionEvaluator=model.predict_proba,
                            initialTemp=100, finalTemp=0.01,
                            tempReduction="linear",
                            iterationPerTemp=10, alpha=10, beta=5)
    SA.run()
