import pandas as pd
import numpy as np
import pickle
from sklearn.svm import OneClassSVM


def get_ocsvm(train_x):
    clf = OneClassSVM(nu=0.0000001).fit(train_x)
    return clf