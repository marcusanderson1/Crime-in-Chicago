#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 08:57:57 2019

@author: marcusanderson
"""
import data_exploration_chart as dec
import model as m
import datetime 

#The Menu options to select when the program is run
order_of_execution = ['1:\tRead in the source data', '2:\tRun preliminary analysis on dataset','3:\tCut down dataset to years 10 to 14',\
                      '4:\tProduce Analysis Charts on Bar/Line charts', '5:\tProduce Heat maps', '6:\tAdd Geographical distance',\
                      '7:\tPre process data','8:\tBuild Logisitic Regression Model','9:\tBuild Neural Network',\
                      '10:\tNearest Neighbours model','11:\tBuild Decision Tree model','12:\tBuild Naive Bayes Model',\
                      '13:\tRead and pre process data for on 15 to 16','14:\tTest Logisitic Regression model on 15 to 16',\
                      '15:\tTest Nearest Neighbours model on 15 to 16','16:\tTest Neural Network model on 15 to 16',\
                      '17:\tTest Decision Tree model on 15 to 16','18:\tTest Naive Bayes model on 15 to 16',\
                      '19:\tLoad models','20:\tQuit']




def call_method(choice):
    #global variables
    global data , data10to14,data15to16, analysis_data, community_area , primary_type , data_enriched_dist  , X,primaryType_onehot,test_data_X , primaryType_onehot_test 
    global list_of_lr_models , list_of_nn_models , list_of_knn_models , list_of_dt_models , list_of_nb_models
    global data10to16 , data15to16_enriched_dist
    if (choice == 1):
        #Reads in the data
        data = dec.read_data(dec.query,dec.conn)
    elif (choice == 2):
        #Runs preliminary analysis and totals
        dec.time_series_graph_month(data)
        dec.time_series_graph_year(data)
        dec.table_primary_type_volume(data)
        dec.table_primary_type_volume05to09(data)
        dec.table_primary_type_volume10to14(data)
    elif (choice == 3):
        #Cuts down the dataset to just the years we are interested
        data10to14, community_area, primary_type = dec.cut_down_data(data)
    elif (choice == 4):
        #production of charts and analysis
        analysis_data = dec.get_analysis(data10to14)
        dec.chart_data(analysis_data, community_area)
        print(data10to14.shape)
        dec.time_series_community_area_graph_year(data10to14,community_area )
        dec.chart_primary_type(data10to14)
        dec.time_series_year_primary_type(data10to14)
        dec.rolling_yearly_primary_type(data, community_area , primary_type) 
        dec.calculate_percentage_arrests(data10to14, community_area)

    elif (choice == 5):
        #production of heat maps
        dec.heatmap_primary_type_location_desc(data10to14)
        dec.heatmap_year_community_area(data10to14)
        dec.heatmap_month_community_area(data10to14)
        dec.heatmap_weekday_community_area(data10to14)
        dec.heatmap_hour_community_area(data10to14)
        dec.heatmap_hour_primary_type(data10to14)
        dec.heatmap_month_primary_type(data10to14)
        dec.heatmap_weekday_primary_type(data10to14)
        
    elif (choice == 6):
        #Enrich the geographic data where the longitude / latitude values are null
        data_enriched = dec.geographical_data_analysis(data10to14,data10to14)
        data_enriched.shape
        data_enriched_dist = dec.calculate_distance_from_mean(data_enriched)
        data_enriched_dist.shape
    elif (choice == 7):
        #create the model data pre processing
        model_data = data_enriched_dist
        model_data= model_data.dropna()#drop the 11 rows containing nulls
        model_data['District'] = model_data.District.astype(float)
        model_data['Ward'] = model_data.Ward.astype(float)
        model_data['Beat'] = model_data.Beat.astype(float)
        communityArea_onehot , month_onehot , weekday_onehot , domestic_onehot , distance_band_onehot , primaryType_onehot , arrest_onehot = m.generate_one_hot_data(model_data)
        X = m.append_one_hot_datasets(model_data,communityArea_onehot , month_onehot , weekday_onehot , distance_band_onehot , arrest_onehot )

    elif (choice == 8):
        #build logisitic regression models
        type_model = 'logisitic_regression_model'
        list_of_lr_models = m.primary_type_model(X,primary_type,primaryType_onehot,type_model)
        m.save_model(list_of_lr_models,type_model)
        m.save_test_results(type_model, list_of_lr_models)
        print(list_of_lr_models)
    elif (choice == 9):
        #build neural network models
        type_model = 'neural_network_model'        
        list_of_nn_models = m.primary_type_model(X,primary_type,primaryType_onehot,type_model)
        m.save_model(list_of_nn_models,type_model)
        m.save_test_results(type_model, list_of_nn_models)
        print(list_of_nn_models)
    elif(choice == 10):
        #build K Nearest Neighbour models
        print('build started', datetime.datetime.now())
        type_model = 'knn_classifier_model'
        list_of_knn_models = m.primary_type_model(X,primary_type,primaryType_onehot,type_model)
        m.save_model(list_of_knn_models,type_model)
        m.save_test_results(type_model, list_of_knn_models)
        print(list_of_knn_models)
        print('build finished', datetime.datetime.now())
    elif(choice == 11):
        #Build Decision Tree models
        type_model = 'decision_tree_model'
        list_of_dt_models = m.primary_type_model(X,primary_type,primaryType_onehot,type_model)
        m.save_model(list_of_dt_models,type_model)      
        m.save_test_results(type_model, list_of_dt_models)
        print(list_of_dt_models)
    elif(choice == 12):
        #Build naive bayes models
        type_model = 'naive_baye_model'
        list_of_nb_models = m.primary_type_model(X,primary_type,primaryType_onehot,type_model)
        m.save_model(list_of_nb_models,type_model)   
        m.save_test_results(type_model, list_of_nb_models)
        print(list_of_nb_models)
    elif (choice == 13):
        #Extract data for 2015/16
        print('Step 1: Get Test Data 2015 and 2016')
        data15to16 = m.fetch_new_data(m.sql, m.conn)

        print('Data read in ',data15to16.shape)
        data15to16 = m.cut_down_test_data(data15to16,  community_area, primary_type)
        print('cut down the dataset',data15to16.shape)
        
        data15to16_enriched = dec.geographical_data_analysis(data15to16,data10to14) 
        print(data15to16_enriched.shape)
        
        print('Add the calculated distances to the dataset')
        data15to16_enriched_dist = dec.calculate_distance_from_mean(data15to16_enriched)
        print('Create the one hot datasets')
        communityArea_onehot_test, month_onehot_test , weekday_onehot_test , domestic_onehot_test , distance_band_onehot_test , primaryType_onehot_test , arrest_onehot_test = m.generate_one_hot_data(data15to16_enriched_dist)
        test_data_X = m.append_one_hot_datasets(data15to16_enriched_dist,communityArea_onehot_test , month_onehot_test , weekday_onehot_test , distance_band_onehot_test , arrest_onehot_test )
        print(len(test_data_X))
        
    elif (choice == 14):
        #Test Logistic Regression model
        print('14 running')
        type_model = 'logisitic_regression_model'
        model_score = m.test_new_data(list_of_lr_models, test_data_X, primary_type, primaryType_onehot_test)
        m.save_results(type_model , model_score)
    elif (choice == 15):
        #Test KNN model
        print('15 running')
        print('Read data started', datetime.datetime.now())
        type_model = 'knn_classifier_model'
        model_score = m.test_new_data(list_of_knn_models, test_data_X, primary_type, primaryType_onehot_test)
        m.save_results(type_model , model_score)
        print('Finished ', datetime.datetime.now())
    elif (choice == 16):
        #Test neural networks
        print('16 running')
        type_model = 'neural_network_model'
        model_score = m.test_new_data(list_of_nn_models, test_data_X, primary_type, primaryType_onehot_test)   
        m.save_results(type_model , model_score)
    elif (choice == 17):
        #Test Decision Tree
        type_model = 'decision_tree_model'
        model_score = m.test_new_data(list_of_dt_models, test_data_X, primary_type, primaryType_onehot_test)
        m.save_results(type_model , model_score)
    elif (choice == 18):
        #Test naive baye model        
        print('you chose 18')
        type_model = 'naive_baye_model'
        model_score = m.test_new_data(list_of_nb_models, test_data_X, primary_type, primaryType_onehot_test)
        m.save_results(type_model , model_score)
    elif (choice == 19):
        #load models from directory
        print('load models')
        
        list_of_lr_models= []
        type_model = 'logisitic_regression_model'
        list_of_lr_models = m.load_model(primary_type,type_model)
        
        list_of_nn_models = []
        type_model = 'neural_network_model'
        list_of_nn_models = m.load_model(primary_type,type_model)      
        
        list_of_knn_models= []
        type_model = 'knn_classifier_model'
        list_of_knn_models = m.load_model(primary_type,type_model)
        
        list_of_dt_models = []
        type_model = 'decision_tree_model'
        list_of_dt_models = m.load_model(primary_type,type_model)    
        
        list_of_nb_models = []
        type_model = 'naive_baye_model'
        list_of_nb_models = m.load_model(primary_type,type_model)        
         
    elif (choice ==0):
        #Read in data for 10 to 16
        data10to16 = m.pre_processing(m.sql_pre_proc, dec.conn, primary_type, community_area )
        #If you are interested in seeing the correlation matrices
        m.feature_selection(data10to16)
    
    elif (choice ==20):
        exit()

    else:
        exit()



if __name__ == '__main__':
    
    
    
    while True:
        
        print('Welcome to my Final Year Project on using a Machine Learning Algorithm to build a \nPredictive model for Crime in Chigcago\n\n')
        for choice in order_of_execution:
            print(choice)

        try:    
            choice_input = int(input('Please choose from the menu above: '))
        except ValueError:
            print('Non numeric value entered, please choose a value from the list')
            #choice_input = None
        
        if (choice_input >= 0 and choice_input < 20):
            call_method(choice_input)
        elif (choice_input == 20):
            break
      
            
        else:
            print('The value entered is not within the menu options, please try again.')
    
        
    
    
    