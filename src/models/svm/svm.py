# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:28:03 2020

@author: Goller
"""

import numpy as np 
from mega import Mega
import os
import zipfile
from zipfile import ZipFile 
import random
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
import itertools
from itertools import cycle

label_dict = {'Animal':0,
              'Humans':1,
              'Natural':2,
             }

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def one_hot_encoder(true_labels, num_records, num_classes):
    temp = np.array(true_labels[:num_records])
    true_labels = np.zeros((num_records, num_classes))
    true_labels[np.arange(num_records), temp] = 1
    return true_labels

def display_results(y_test, pred_probs, cm = True):
    pred = np.argmax(pred_probs, axis=-1)
    one_hot_true = one_hot_encoder(y_test, len(pred), len(label_dict))
    print('Test Set Accuracy =  {0:.2f}'.format(accuracy_score(y_test, pred)))
    print('Test Set F-score =  {0:.2f}'.format(f1_score(y_test, pred, average='macro')))
    #print('ROC AUC = {0:.3f}'.format(roc_auc_score(y_true=one_hot_true, y_score=pred_probs, average='macro')))
    

os.chdir(r'C:\Users\Goller\Desktop\audio-event-tagging\src\utils')
import load_file as ld

path=r"C:\Users\Goller\Desktop\audio-event-tagging\data\Audio"
os.chdir(path)

X_train,X_test,y_train,y_test=ld.get_data(path)

X_train=np.concatenate( X_train, axis=0 )
X_test=np.concatenate(X_test,axis=0)

# Train
svm_classifier = SVC(C=10000.0, probability = True, kernel='rbf')
svm_classifier.fit(X_train, y_train)

# Predict
pred_probs = svm_classifier.predict_proba(X_test)

# Results
display_results(y_test, pred_probs)


