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

def get_config():
    config = configparser.ConfigParser()
    #config.read(sys.argv[1])
    config.read('configurations.txt')
    config = config['DEFAULT']
    return config

def get_constrains(dataset_name, perturbability_path):
    perturbability = pd.read_csv(perturbability_path)
    perturbability = perturbability["Perturbability"].to_numpy()

    if dataset_name == "RADCOM":
        constrains = {
            'agg_count':
            'delta_delta_delta_from_previous_request'
            'delta_delta_from_previous_request'
            'delta_from_previous_request'
            'delta_from_start'
            'effective_peak_duration'
            'index'
            'minimal_bit_rate'
            'non_request_data'
            'peak_duration'
            'peak_duration_sum'
            'previous_previous_previous_previous_total_sum_of_data_to_sec'
            'previous_previous_previous_total_sum_of_data_to_sec'
            'previous_previous_total_sum_of_data_to_sec'
            'previous_total_sum_of_data_to_sec'
            'sum_of_data'
            'total_sum_of_data'
            'total_sum_of_data_to_sec'
            'serv_label'
            'start_of_peak_date'
            'start_of_peak_time'
            'end_of_peak_date'
            'end_of_peak_time'
            }
        feature_range = {
            'agg_count'
            'delta_delta_delta_from_previous_request'
            'delta_delta_from_previous_request'
            'delta_from_previous_request'
            'delta_from_start'
            'effective_peak_duration'
            'index'
            'minimal_bit_rate'
            'non_request_data'
            'peak_duration'
            'peak_duration_sum'
            'previous_previous_previous_previous_total_sum_of_data_to_sec'
            'previous_previous_previous_total_sum_of_data_to_sec'
            'previous_previous_total_sum_of_data_to_sec'
            'previous_total_sum_of_data_to_sec'
            'sum_of_data'
            'total_sum_of_data'
            'total_sum_of_data_to_sec'
            'serv_label'
            'start_of_peak_date'
            'start_of_peak_time'
            'end_of_peak_date'
            'end_of_peak_time'
        }
            

    return constrains, perturbability, feature_range


class SimulatedAnnealing:
    def __init__(self, initialSolution, solutionEvaluator, initialTemp, finalTemp, tempReduction, neighborOperator,
                 iterationPerTemp=100, alpha=10, beta=5):
        self.solution = initialSolution
        self.evaluate = solutionEvaluator
        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        self.iterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.beta = beta
        self.neighborOperator = neighborOperator

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

    def run(self):
        while not self.isTerminationCriteriaMet():
            # iterate that number of times, based on the temperature
            for i in range(self.iterationPerTemp):
                # get all the neighbors
                neighbors = self.neighborOperator(self.solution)
                # pick a random neighbor
                newSolution = random.choice(neighbors)
                # get the cost between the two solutions
                cost = self.evaluate(self.solution) - self.evaluate(newSolution)
                # if the new solution is better, accept it
                if cost >= 0:
                    self.solution = newSolution
                # if the new solution is not better, accept it with a probability of e^(-cost/temp)
                else:
                    if random.uniform(0, 1) < math.exp(-cost / self.currTemp):
                        self.solution = newSolution
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
    record = x_attack.iloc[1]
    record_pred = y_attack.iloc[1]
    # prob_loc = np.abs(1 - record_pred)

    constrains, perturbability, feature_range = get_constrains(dataset_name, perturbability_path)
   
    def neighbor_operator(current):
    # return all neighbor of cuurent
    # neighbor is a sample that differ from current in one edittible feature
        edittible = perturbability
        neighbors = []
        for feature in edittible.Row:
            if (edittible.perturbability[feature] == 1): # can be edit
                for change_range in feature_range:
                    neighbor = current.copy()
                    if (neighbor[feature] != change_range):
                        neighbor[feature] = change_range
                        neighbors.append()
        
        return neighbors


    model = pickle.load(open('Models/RADCOM/RADCOM_target_RF_seed-42_estimators-500_maxdepth-9.pkl', 'rb'))
    print(model.predict_proba(record.values.reshape(1, -1))[0][int(record_pred)])

    path_cost = model.predict_proba(record.values.reshape(1, -1))[0][int(record_pred)]
    

    SA = SimulatedAnnealing(initialSolution=init_path, solutionEvaluator=path_cost, initialTemp=100, finalTemp=0.01,
                             tempReduction="linear", neighborOperator=neighbor_operator, iterationPerTemp=100, alpha=10,
                             beta=5)
