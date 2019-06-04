#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:46:34 2019

@author: marcusanderson
"""

#import of libraries

import csv
import pymysql
import numpy as np
import pandas as pd
import queries as q
import db_connection as db
import matplotlib.pyplot as plt
import datetime 
from sklearn import model_selection, linear_model

from sklearn.preprocessing import RobustScaler , StandardScaler, normalize

import pickle
import seaborn as sns
from sklearn.metrics import accuracy_score  , confusion_matrix  , precision_score, recall_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


conn = db.open_sql_connection()
sql = q.sql_new_data
sql_pre_proc = q.sql_data_pre_processing




def pre_processing(query_pre_proc, conn , primary_type, community_area ):
    print('get data from 10 to 16')
    print('slice the data by community area/primary type')
    print('factorize categorical variables')
    data_pre_proc = pd.read_sql(query_pre_proc , conn)
    
    data_pre_proc = data_pre_proc[data_pre_proc['PrimaryType'].isin(primary_type)]
    data_pre_proc = data_pre_proc[data_pre_proc['CommunityArea'].isin(community_area)]
    
    #data_pre_proc.Date = pd.to_datetime(data_pre_proc.Date,format='%m/%d/%Y %I:%M:%S %p')
    #data_pre_proc.index = pd.DatetimeIndex(data_pre_proc.Date)
    data_pre_proc['District'] = data_pre_proc.District.astype(float)
    data_pre_proc['Ward'] = data_pre_proc.Ward.astype(float)
    data_pre_proc['Beat'] = data_pre_proc.Beat.astype(float)
    
    #factorize categorical variables
    #data_pre_proc['Block'] = pd.factorize(data_pre_proc['Block'])[0]
    
    
    #data_pre_proc['IUCR'] = pd.factorize(data_pre_proc['IUCR'])[0]
    #data_pre_proc['Description'] = pd.factorize(data_pre_proc['Description'])[0]
    
    #data_pre_proc['LocationDescription'] = pd.factorize(data_pre_proc['LocationDescription'])[0]
    #data_pre_proc['FBICode'] = pd.factorize(data_pre_proc['FBICode'])[0]
    #data_pre_proc['Location'] = pd.factorize(data_pre_proc['Location'])[0]
    data_pre_proc['CommunityArea'] = pd.factorize(data_pre_proc['CommunityArea'])[0] 
    data_pre_proc['PrimaryType'] = pd.factorize(data_pre_proc['PrimaryType'])[0] 
    data_pre_proc['Arrest'] = pd.factorize(data_pre_proc['Arrest'])[0] 
    data_pre_proc['Domestic'] = pd.factorize(data_pre_proc['Domestic'])[0]
    data_pre_proc['Month'] = pd.factorize(data_pre_proc['Month'])[0]
    data_pre_proc['Weekday'] = pd.factorize(data_pre_proc['Weekday'])[0]
    
    data_pre_proc.info()
    data_pre_proc = data_pre_proc.drop(['CaseNumber','Date'], axis =1)
    
    #data_factorized = data_pre_proc[data_pre_proc['ID','Block','IUCR','Description','LocationDescription','FBICode','Location']]
    #data_factorized.to_csv('factorize.csv')
    return data_pre_proc



def feature_selection(data):
    #Give us our target features
    years = ['2010','2011','2012','2013','2014']
    data = data[data['Year'].isin(years)]
    
    #Correlation
    plt.figure(figsize=(20,10))
    cor = data.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.YlGnBu)
    plt.show()
    
    cor_target = abs(cor['PrimaryType'])
    relevant_features = cor_target[cor_target>0.1]
    #strong correlation
    print(relevant_features)


def fetch_new_data(query, conn):
    print('Read data started', datetime.datetime.now())
    data = pd.read_sql(query , conn)
    #CREATE DATE INDEX
    print (query)
    print('data read',data.shape)
    
    data.Date = pd.to_datetime(data.Date,format='%m/%d/%Y %I:%M:%S %p')
    data.index = pd.DatetimeIndex(data.Date)
    data['District'] = data.District.astype(float)
    data['Ward'] = data.Ward.astype(float)
    data['Beat'] = data.Beat.astype(float)
    
    print('Read data finished', datetime.datetime.now())
    return data

def cut_down_test_data(data15to16 , community_area, primary_type):

    data = data15to16[data15to16['PrimaryType'].isin(primary_type)]
    data = data15to16[data15to16['CommunityArea'].isin(community_area)]
    
    return data

def generate_one_hot_data(data):
    #Creation of One hot datasets
    
    communityArea_onehot = pd.get_dummies(data['CommunityArea'])
    month_onehot = pd.get_dummies(data['Month'])
    weekday_onehot = pd.get_dummies(data['Weekday'])
    domestic_onehot = pd.get_dummies(data['Domestic'])
    #location_desc = pd.get_dummies(data['LocationDescription'])
    distance_band_onehot = pd.get_dummies(data['DistanceBand'])
    primaryType_onehot  = pd.get_dummies(data['PrimaryType'])
    
    domestic_onehot = pd.get_dummies(data['Domestic'])
    arrest_onehot  = pd.get_dummies(data['Arrest'])    
    return communityArea_onehot , month_onehot , weekday_onehot , domestic_onehot , distance_band_onehot , primaryType_onehot, arrest_onehot

def append_one_hot_datasets(data,communityArea_onehot , month_onehot , weekday_onehot , distance_band_onehot , arrest_onehot ):

    
    #Here is where we appended the one hot datasets to the features to create the X numpy array

    X = np.array(data[['LocationDescription','Description','IUCR','FBICode']])#because these are numeric variables no need to split columns into 0 and 1
    X = np.hstack([X, communityArea_onehot, arrest_onehot])


    return X
    


def build_logisitic_regression_model( primaryType, X,primaryType_onehot):
    
    #Here we construct the loigistic regression model
    
    robust_scaler = RobustScaler()
      
    y = np.array(primaryType_onehot[primaryType])
    print('len X', len(X), ' len y', len(y))  
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = .20, random_state=42)
    
    X_train = robust_scaler.fit_transform(X_train)
    X_test = robust_scaler.fit_transform(X_test)
    print('len X train', len(X_train), ' len X test', len(X_test))  
    print('len y train', len(y_train), ' len y test', len(y_test))    
    
    model = linear_model.LogisticRegression(solver='lbfgs',max_iter=100)
    model.fit(X_train, y_train)
    print('ln 163')
    
    y_score = model.score(X_test,y_test)#accuracy score
    print('ln 165')    
    y_pred = model.predict(X_test)#predictions
    print('ln 167')    
    precision = precision_score(y_test,y_pred)#precision score
    recall = recall_score(y_test,y_pred)#recall score
    
    true_negative ,false_positive , false_negative,true_positive  = confusion_matrix(y_test , y_pred).ravel()#output of confusion matrix

    return model , primaryType , y_score , true_positive , false_positive , true_negative , false_negative , precision , recall


def build_neural_network_model(primaryType, X,primaryType_onehot):
    #Creates Neural Network models
    scaler = StandardScaler()
    
    y = np.array(primaryType_onehot[primaryType])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = .20, random_state=42)
    
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)

    
    model_nn = MLPClassifier(hidden_layer_sizes=(13,13,13),activation='logistic',max_iter=500)
    model_nn.fit(X_train,y_train)
    y_score = model_nn.score(X_test, y_test)
    
    y_pred = model_nn.predict(X_test)
    
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    
    true_negative ,false_positive , false_negative,true_positive  = confusion_matrix(y_test , y_pred).ravel()
    
    #print(model_nn.predict(X_test))
    return model_nn , primaryType , y_score , true_positive , false_positive , true_negative , false_negative , precision , recall
    
def build_knn_classifier_model(primaryType, X,primaryType_onehot):
    #Create nearest neighbour model
 
    y = np.array(primaryType_onehot[primaryType])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = .20, random_state=42)
    #Scale dataset
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_train,y_train ) #Fit models with features
    y_score = model_knn.score(X_test, y_test)#accuracy score
    
    y_pred = model_knn.predict(X_test)#prdictions
    precision = precision_score(y_test,y_pred)#presicion
    recall = recall_score(y_test,y_pred)#recall score
    
    true_negative ,false_positive , false_negative,true_positive  = confusion_matrix(y_test , y_pred).ravel() #confusion matrix   
    
    return model_knn , primaryType , y_score, true_positive , false_positive , true_negative , false_negative , precision , recall
    
def build_decision_tree_model(primaryType, X,primaryType_onehot):
    '''
    Build decision tree model
    '''
 
    y = np.array(primaryType_onehot[primaryType])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = .20, random_state=42)


    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    model_dt = DecisionTreeClassifier()
    model_dt.fit(X_train,y_train )
    
    y_score = model_dt.score(X_test , y_test)#accuracy

    y_pred = model_dt.predict(X_test)#predictions
    precision = precision_score(y_test,y_pred)#precision
    recall = recall_score(y_test,y_pred)
    
    true_negative ,false_positive , false_negative,true_positive  = confusion_matrix(y_test , y_pred).ravel()       #confusion matrix 
    
    return model_dt , primaryType , y_score , true_positive , false_positive , true_negative , false_negative , precision , recall
    
def build_naive_baye_model(primaryType, X,primaryType_onehot):
    
    y = np.array(primaryType_onehot[primaryType])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = .20, random_state=42)
    
    scaler = StandardScaler()
    
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    
    y_score = model_nb.score(X_test, y_test)

    y_pred = model_nb.predict(X_test)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    
    true_negative ,false_positive , false_negative,true_positive  = confusion_matrix(y_test , y_pred).ravel()       
    
    return model_nb, primaryType , y_score, true_positive , false_positive , true_negative , false_negative , precision , recall
    

def primary_type_model(X, primary_type,primaryType_onehot, type_model):
    
    list_of_models = []
    
    
    for p in primary_type:
        if (type_model == 'logisitic_regression_model'):    
            model , ptype,score, true_positive , false_positive , true_negative , false_negative , precision , recall = build_logisitic_regression_model(p, X, primaryType_onehot)
        if (type_model == 'neural_network_model'):    
            model , ptype, score , true_positive , false_positive , true_negative , false_negative , precision , recall = build_neural_network_model(p, X, primaryType_onehot)
        if(type_model == 'knn_classifier_model'):
            model , ptype, score, true_positive , false_positive , true_negative , false_negative , precision , recall = build_knn_classifier_model(p, X, primaryType_onehot)
        if(type_model=='decision_tree_model'):
            model , ptype, score, true_positive , false_positive , true_negative , false_negative , precision , recall = build_decision_tree_model(p, X, primaryType_onehot)  
        if (type_model=='naive_baye_model'):
            model , ptype, score, true_positive , false_positive , true_negative , false_negative , precision , recall = build_naive_baye_model(p, X, primaryType_onehot)
        
        list_of_models.append([model,ptype,score, true_positive , false_positive , true_negative , false_negative , precision , recall])
        
    return list_of_models


def test_new_data(list_of_models, test_data_X, primary_type, primaryType_onehot_test):
    '''
    
    Reads in the models
    reads in the test data
    Produces statisticts for predictions, recall, precision
    '''
    print(list_of_models)
    len(list_of_models)
    
    model_score = []

    for i in range(len(list_of_models)):
    
        model = list_of_models[i][0]
        model_type = list_of_models[i][1]
    
        y = np.array(primaryType_onehot_test[model_type])

        y_pred = model.predict(test_data_X)

        y_score = accuracy_score(y, y_pred, normalize=True, sample_weight=None)
        
        precision = precision_score(y,y_pred)
        recall = recall_score(y,y_pred)
        
        true_negative ,false_positive , false_negative,true_positive  = confusion_matrix(y , y_pred).ravel()
        
        
        '''
        confusion_matrices = confusion_matrix(y , y_pred)
        true_positive = confusion_matrices[0][0]
        false_positive = confusion_matrices[0][1]
        true_negative = confusion_matrices[1][0]
        false_negative = confusion_matrices[1][1]
        '''
        model_score.append([model_type ,y_score , true_positive , false_positive , true_negative, false_negative, precision, recall])
    
    return model_score

def save_results(type_model , model_score):
    #save test results against test data
    results = pd.DataFrame(model_score)
    results.to_csv('Results/'+type_model+'.csv')

def save_test_results(type_model, list_of_models):
    #save validation results of the models
    test_model_score = []
    for m in range(len(list_of_models)):
        primaryType = list_of_models [m][1]
        score = list_of_models [m][2]
        tp = list_of_models [m][3]
        fp = list_of_models [m][4]
        tn = list_of_models [m][5]
        fn = list_of_models [m][6]
        precision = list_of_models [m][7]
        recall = list_of_models [m][8]
        
        test_model_score.append([primaryType , score , tp, fp, tn, fn, precision, recall])
    
    results = pd.DataFrame(test_model_score)
    results.to_csv('Results/Test/'+type_model+'.csv')    
        
def save_model(list_of_models,type_model):
    #save models to file directory
    for a in range(len(list_of_models)):
        model_primary_type = list_of_models[a][1]
        model = list_of_models[a][0]
        if (type_model == 'logisitic_regression_model'):
            path = 'Models/LogisticRegression/'
            filename = type_model +'_' +model_primary_type+'.sav'
        if (type_model == 'neural_network_model'):    
            path = 'Models/NeuralNetwork/'
            filename = type_model +'_' +model_primary_type+'.sav'
        if(type_model == 'knn_classifier_model'):
            path = 'Models/NearestNeighbour/'
            filename = type_model +'_' +model_primary_type+'.sav'
        if(type_model=='decision_tree_model'):
            path = 'Models/DecisionTree/'
            filename = type_model +'_' +model_primary_type+'.sav' 
        if (type_model=='naive_baye_model'):
            path = 'Models/NaiveBayes/'
            filename = type_model +'_' +model_primary_type+'.sav' 
        pickle.dump(model,open(path+filename,'wb'))

def load_model(primary_type,type_model):
    #load models from file directory
    loaded_models = []
    for pt in primary_type:
        if (type_model == 'logisitic_regression_model'):
            path = 'Models/LogisticRegression/'
            filename = type_model +'_' +pt+'.sav'
        if (type_model == 'neural_network_model'):    
            path = 'Models/NeuralNetwork/'
            filename = type_model +'_' +pt+'.sav'
        if(type_model == 'knn_classifier_model'):
            path = 'Models/NearestNeighbour/'
            filename = type_model +'_' +pt+'.sav'
        if(type_model=='decision_tree_model'):
            path = 'Models/DecisionTree/'
            filename = type_model +'_' +pt+'.sav' 
        if (type_model=='naive_baye_model'):
            path = 'Models/NaiveBayes/'
            filename = type_model +'_' +pt+'.sav' 
        
        get_model = pickle.load(open(filename,'rb'))
        loaded_models.append([get_model , pt])
        
    return loaded_models