#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:45:52 2019

@author: shubham
"""
from data_gathering import Data_gathering
import pickle
import pandas as pd

class Model_Pred:
    """ 
    This is a class for Feature engineering in order to derive new feature
    and also help with imputation. 
      
    Attributes: 
         
    """        
    def run_test_prediction(self,pickled_model:bytes,test_data:pd.DataFrame):
         
        # Load the pickled model 
        model = pickle.loads(pickled_model)        
        output = model.predict(test_data).astype(int)
        df_output = pd.DataFrame()
        aux = Data_gathering().load_dataset(file_path=r'/home/shubham/Desktop/open_projects/',file_name='test.csv')
        df_output['PassengerId'] = aux['PassengerId']
        df_output['Survived'] = output
        df_output[['PassengerId','Survived']].to_csv(r'/home/shubham/Desktop/open_projects/pred.csv',index=False)
        