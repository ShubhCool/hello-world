#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:45:52 2019

@author: shubham
"""
from codes.data_gathering import Data_gathering
from codes.feature_engineering import Feature_Engineering
import pickle
import pandas as pd
import os

class Model_Pred:
    """ 
    This is a class for Feature engineering in order to derive new feature
    and also help with imputation. 
      
    Attributes: 
         
    """        
    def run_test_prediction(self,test_data:pd.DataFrame,save_path:str,pkl_name:str,scaling_rqd:bool):
         
        # Load the pickled model 
        model = pickle.load(open(os.path.join(save_path,pkl_name),'rb'))
        if scaling_rqd==True:
            test_data=Feature_Engineering().input_feat_scale(input_data=test_data,train_yes=False)            
        output = model.predict(test_data).astype(int)
        df_output = pd.DataFrame()
        aux = Data_gathering().load_dataset(file_path=r'/home/shubham/Desktop/open_projects/data/',file_name='test.csv')
        df_output['PassengerId'] = aux['PassengerId']
        df_output['Survived'] = output
        df_output[['PassengerId','Survived']].to_csv(r'/home/shubham/Desktop/open_projects/pred.csv',index=False)
        
    
    def save_model_pickle(self,byte_model:bytes,save_path:str,pkl_name:str):
        
        model= pickle.loads(byte_model)
        
        with open(os.path.join(save_path,pkl_name), 'wb') as f:
            pickle.dump(model, f)
          
   
            


        
        
        