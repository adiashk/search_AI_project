"""
Simulated Annealing Class
"""
import pickle
import random
import math
import sklearn
import pandas as pd


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
            # iterate that number of times
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

df = pd.read_csv('Datasets/RADCOM/after_preprocessing/RADCOM_after_preprocessing.csv')
df = df.drop(columns=['pred'])
record = df.iloc[1]
model = pickle.load(open('Models/RADCOM/RADCOM_target_RF_seed-42_estimators-500_maxdepth-9.pkl', 'rb'))
print(model.predict(record.values.reshape(1, -1))[0])
# SA = SimulatedAnnealing(initialSolution=init_path, solutionEvaluator=path_cost, initialTemp=100, finalTemp=0.01,
#                         tempReduction="linear", neighborOperator=neighbor_operator, iterationPerTemp=100, alpha=10,
#                         beta=5)
