# -*- coding: utf-8 -*-
import numpy as np 
import os
import zipfile
from sklearn.model_selection import train_test_split
$ from sklearn.model_selection import train_test_split
#from zipfile import ZipFile 
#import random
#import numpy as np
#import pandas as pd


#path=r"C:\Users\Goller\Desktop\audio-event-tagging\data\Audio"


label_dict = {'Animal':0,
              'Humans':1,
              'Natural':2,
             }

def get_data(path,i=1,test_dim=0.2,zip_name="features_.zip"):
    #path=path of the zip file
    
    os.chdir(path)
    #upload the file
    file_name = zip_name
    zf = zipfile.ZipFile(file_name, 'r')
    
    if i==1:
        method='mfcc'
    else:
        method='mel_spec'
    
    #Create dataset
    X=[np.load(zf.open(nam),allow_pickle =True).tolist()[method].reshape(1,-1) for nam in zf.namelist()]
    y=[label_dict[nam.split('_')[1].split('.')[0]] for nam in zf.namelist()]
    
    #Create Test, Train datatset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_dim, random_state=42)
    
    #Fix dataset
    #Train Dataset
    a=0
    for i in range(len(X_train)):
        if len(X_train[a][0])>5602:
            X_train[a]=X_train[a][:,:5603]
        else:# len(X_train[a][0])<5603:
            #print(len(X_train[a][0]))
            X_train.pop(a)
            y_train.pop(a)
            a=a-1
        a=a+1
    print('Fine primo ciclo')
    #Test Dataset
    a=0
    for i in range(len(X_test)):
        if len(X_test[a][0])>5602:
            X_test[a]=X_test[a][:,:5603]
        else:# len(X_train[a][0])<5603:
            #print(len(X_test[a][0]))
            X_test.pop(a)
            y_test.pop(a)
            a=a-1
        a=a+1
    print('Fine Secondo ciclo')

    X_train=np.concatenate( X_train, axis=0 )
    X_test=np.concatenate(X_test,axis=0)
    return X_train,X_test,y_train,y_test

#X_train,X_test,y_train,y_test=get_data(path)
