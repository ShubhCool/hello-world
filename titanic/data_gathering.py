#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:42:00 2019

@author: shubham
"""
import pandas as pd
import os

class Data_gathering:
    """ 
    This is a class for data_gathering. 
      
    Attributes: 
        raw_data(pd.DataFrame): pandas dataframe 
    """
    def __init__(self):
        '''
        Initalization of Data_gathering constructor
        self:object
        '''
        self.raw_data = pd.DataFrame({})
        
    def load_dataset(self, file_path: str,file_name: str) -> pd.DataFrame:
        """ 
        The function to load pandas dataframe. 
  
        Parameters: 
            file_path(str):file path
            file_name(str): file name
          
        Returns: 
            pd.DataFrame: pandas dataframe. 
        """
        combined_path=os.path.join(file_path,file_name)
        self.raw_data=pd.read_csv(combined_path)
        return self.raw_data
    
    def get_combined_data(self, file_path: str, train_file_name: str,
                          test_file_name: str) -> pd.DataFrame:
        """ 
        The function to combine train and test pandas dataframe. 
  
        Parameters: 
            file_path(str):file path
            train_file_name(str): training data file name
            test_file_name(str): test data file name
          
        Returns: 
            pd.DataFrame: train and test combined pandas dataframe. 
        """
        train_data=self.load_dataset(file_path,train_file_name)
        train_data=train_data.drop('Survived', 1)
        test_data=self.load_dataset(file_path,test_file_name)

        combined_data = train_data.append(test_data)
        combined_data.reset_index(inplace=True)
        combined_data.drop('index', inplace=True, axis=1)

        return combined_data


