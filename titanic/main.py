#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 07:26:23 2019

@author: shubham
"""
from data_gathering import Data_gathering
from feature_prepration import Feature_Prepration
from feature_engineering import Feature_Engineering
from Model_Run import Model_manager
from Model_prediction import Model_Pred
import timeit
import pickle
'''
#one time activity
import sys
sys.path.append('/home/shubham/Desktop/open_projects/')
'''

if __name__ == '__main__':
    
    DG=Data_gathering()
    train_data = DG.load_dataset(file_path=r'/home/shubham/Desktop/open_projects/',file_name='train.csv')
    #test_data = DG.load_dataset(file_path=r'/home/shubham/Desktop/open_projects/',file_name='test.csv')
    combined_data=DG.get_combined_data(file_path=r'/home/shubham/Desktop/open_projects/',train_file_name='train.csv',test_file_name='test.csv')
    
    FE=Feature_Engineering(combined_data)
    combined_data=FE.derive_new_cols()
    
    FP=Feature_Prepration(combined_data)
    FP.separate_variable_types()
    FP.missing_columns_check()
    combined_data=FP.missing_values_imputation()
    
    combined_data=FE.dummy_encoding(combined_data,col_list=['Pclass','Embarked','Title',
                                'Cabin','Sex'])
    
    targets = train_data.Survived
    train_data = combined_data[ :891]
    test_data=combined_data[891:]
    
    FS_model=pickle.loads(Model_manager().Tree_Based_FS(train_data,targets))
    train_reduced=FS_model.transform(train_data)
    test_reduced=FS_model.transform(test_data)
    
    MM=Model_manager()
    start = timeit.timeit()
    model=MM.rf_model_train(train_reduced,targets,run_gs=False)
    #MM.gbm_model_train(train_data,targets,run_gs=True)
    end = timeit.timeit()
    print('time taken to run this model',(end - start))
    
    MP=Model_Pred()
    MP.run_test_prediction(model,test_reduced)
    


    
    
    
    
    
    