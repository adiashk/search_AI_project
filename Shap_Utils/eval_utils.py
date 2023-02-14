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


# Visualization
import matplotlib.pyplot as plt
import seaborn as sn

#evaluation
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, PrecisionRecallDisplay


def plot_confusion_matrix_to_screen(y, y_preds, title):
    conf_mat = confusion_matrix(y, y_preds, [0, 1])

    cmn = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    cmn *= 100  # for percentages

    print(title)
    print(cmn)

    #return auc, f1, acc
    
def eval_models(model, x_train, y_train, x_test, y_test, type=None):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)


    if (type == 'nn'):

        prob = model.predict_proba(x_test)
        predictions = model.predict(x_test)
        predictions_th = np.where(model.predict_proba(x_test)> 0.5, 1, 0)
        if (predictions_th == predictions):
            print ('preds equals')

        # accuracy on training set
        accuracy_score(y_test, predictions)
        target_names=['adv', 'benign']
        labels=[1, 0]
        print(classification_report(y_test, predictions, target_names=target_names))
        
        preds_train = model.predict(x_train)
        score_train = roc_auc_score(y_train, preds_train)
        preds_val = model.predict(x_test)
        score_val = roc_auc_score(y_test, preds_val)

        print('Train ROC-AUC: {:.2f}'.format(score_train))
        print('Validation ROC-AUC: {:.2f}'.format(score_val))
        
        preds_val = np.where(model.predict_proba(x_test)[:,0]> 0.5, 1, 0)
        val_pred_proba =  model.predict_proba(x_test)
    
        #fpr, tpr, _ = roc_curve(y_test, val_pred_proba[:,0])
        #auc_test = auc(fpr, tpr)
        accuracy_test = accuracy_score(y_test, preds_val)
        f1_test = f1_score(y_test, preds_val, pos_label=1)
        print("Validation Accuracy   %0.4f" % (accuracy_test))
        print("Validation F1-Score    %0.4f" % (f1_test))
        #print("Validation AUC    %0.4f" % (auc_test))

        # Efrat: we print it this way because it seems much better than the visualize graph
        plot_confusion_matrix_to_screen(y_train, model.predict_proba(x_train)[:,0] > 0.5,
                                            title='Surr NN ' + ' Train Set Confusion Matrix (End2End)')
        plot_confusion_matrix_to_screen(y_test, model.predict_proba(x_test)[:,0] > 0.5,
                                            title='Surr NN ' + ' Validation Set Confusion Matrix (End2End)')
        return  round(f1_test,3), round(accuracy_test,3) #round(auc_test,3),
    
    else: ## trees
        try:
            #output = model(x_test)    
            #softmax = torch.exp(output).cpu()
            #prob = list(softmax.numpy())
            #predictions = np.argmax(prob, axis=1)
            prob_train = model.predict_proba(np.array(x_train))
            predictions_train = model.predict(np.array(x_train))
            prob_test = model.predict_proba(np.array(x_test))
            predictions_test = model.predict(np.array(x_test))

            # accuracy on training set
            acc_test = accuracy_score(y_test, predictions_test)
            target_names=['adv', 'benign']
            labels=[1, 0]
            print(classification_report(y_test, predictions_test, target_names=target_names))

            score_train = roc_auc_score(y_train, prob_train[:, 1])
            score_val = roc_auc_score(y_test, prob_test[:, 1])

            print('Train ROC-AUC: {:.2f}'.format(score_train))
            print('Validation ROC-AUC: {:.2f}'.format(score_val))
        except:
            pass

        y_pred_train = model.predict(np.array(x_train))
        y_pred_test = model.predict(np.array(x_test))
        y_pred_proba_test = model.predict_proba(np.array(x_test))[:, 1].flatten()
        y_true_test = np.array(y_test, dtype=np.float64)
        y_true_train = np.array(y_train, dtype=np.float64)

        #fpr, tpr, _ = roc_curve(y_true, y_pred_proba_test)
        #auc_test = auc(fpr, tpr)
        accuracy_train = accuracy_score(y_true_train, y_pred_train)
        f1_train = f1_score(y_true_train, y_pred_train, pos_label=1)
        print("Train Accuracy   %0.4f" % (accuracy_train))
        print("Train F1-Score    %0.4f" % (f1_train))

        accuracy_test = accuracy_score(y_true_test, y_pred_test)
        f1_test = f1_score(y_true_test, y_pred_test, pos_label=1)
        print("Validation Accuracy   %0.4f" % (accuracy_test))
        print("Validation F1-Score    %0.4f" % (f1_test))

        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test, pos_label=1)
        print("Validation Accuracy   %0.4f" % (accuracy_test))
        print("Validation F1-Score    %0.4f" % (f1_test))
        #print("Validation AUC    %0.4f" % (auc_test))

        # Efrat: we print it this way because it seems much better than the visualize graph
        plot_confusion_matrix_to_screen(y_train, y_pred_train, title=type + " Train Confusion Matrix")
        plot_confusion_matrix_to_screen(y_test, y_pred_test, title=type + " Validation Confusion Matrix")

        #_, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 5))
        #plot_confusion_matrix(y_train, y_pred_train, axes[0], title=model_type + " Train Confusion Matrix")
        #plot_confusion_matrix(y_true, y_pred, axes[1], title=model_type + " Validation Confusion Matrix")
        #plt.tight_layout()
        #plt.show()
        return    round(f1_test,3), round(accuracy_test,3) #round(auc_test,3),
