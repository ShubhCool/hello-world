#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:46:09 2019

@author: shubham
"""
import pandas as pd
import numpy as np

class Feature_Prepration:
    """ 
    This is a class for Feature prepration in order to deal with 
    outliers and missing values. 
      
    Attributes: 
         
    """
    def __init__(self, raw_data: pd.DataFrame):
        '''
        Initalization of class object and it's attributes
        Parameters: 
            self: pandas dataframe
        Returns: 
            None
        '''
        self.raw_data = raw_data
        self.continuous = None
        self.categorical = None
        self.discrete = None
        self.encoding_dict = {}
        self.missing_values_cols={}
        
    def separate_variable_types(self) -> None:
        '''
        The function to list out the different variable types. 
  
        Parameters: 
            self: class object
        Returns: 
            None
        '''
        # find categorical variables
        # this should be done based off training data, i.e., self.raw_data
        self.categorical = [var for var in self.raw_data.columns
            if self.raw_data[var].dtype == 'O'
        ]
        #print('There are {} categorical variables'.format(len(self.categorical)))

        # find numerical variables
        # this should be done based off training data, i.e., self.raw_data
        numerical = [var for var in self.raw_data.columns
                     if self.raw_data[var].dtype != 'O']
        #print('There are {} numerical variables'.format(len(numerical)))

        # find discrete variables
        # this should be done based off training data, i.e., self.raw_data
        self.discrete = []
        for var in numerical:
            if len(self.raw_data[var].unique()) < 20 :
                self.discrete.append(var)

        #print('There are {} discrete variables'.format(len(self.discrete)))

        self.continuous = [
            var for var in numerical if
            var not in self.discrete and var not in ['PassengerId']
        ]
            
    def missing_columns_check(self) ->None:
        '''
        The function to list out the columns list whose have missing values. 
        Parameters: 
        self: class object
        Returns: None    
        '''
        
        for col in self.raw_data.columns:
            if self.raw_data.loc[:, (col)].isnull().sum() > 0:
                self.missing_values_cols[col]=self.raw_data.loc[:, (col)].isnull().sum()
        #print("Missing values of these columns {}",format(self.missing_values_cols))        
     
    def missing_values_imputation(self) ->pd.DataFrame:
        '''
        The function to list out the columns list whose have missing values. 
        Parameters: 
        self: class object
        Returns: None    
        '''
        
        # to impute age column (can be changed)
        input_grouped = self.raw_data.groupby(['Sex','Pclass','Title'])
        input_grouped_median = input_grouped.median()
      
        for col in self.missing_values_cols.keys():
            
            if col =='Age':
                def fillAges(row, grouped_median):
                    if row['Sex']=='female' and row['Pclass'] == 1:
                        if row['Title'] == 'Miss':
                            return grouped_median.loc['female', 1, 'Miss']['Age']
                        elif row['Title'] == 'Mrs':
                            return grouped_median.loc['female', 1, 'Mrs']['Age']
                        elif row['Title'] == 'Officer':
                            return grouped_median.loc['female', 1, 'Officer']['Age']
                        elif row['Title'] == 'Royalty':
                            return grouped_median.loc['female', 1, 'Royalty']['Age']
                        
                    elif row['Sex']=='female' and row['Pclass'] == 2:
                        if row['Title'] == 'Miss':
                            return grouped_median.loc['female', 2, 'Miss']['Age']
                        elif row['Title'] == 'Mrs':
                            return grouped_median.loc['female', 2, 'Mrs']['Age']
                            
                    elif row['Sex']=='female' and row['Pclass'] == 3:
                        if row['Title'] == 'Miss':
                            return grouped_median.loc['female', 3, 'Miss']['Age']
                        elif row['Title'] == 'Mrs':
                            return grouped_median.loc['female', 3, 'Mrs']['Age']

                    elif row['Sex']=='male' and row['Pclass'] == 1:
                        if row['Title'] == 'Master':
                            return grouped_median.loc['male', 1, 'Master']['Age']
                        elif row['Title'] == 'Mr':
                            return grouped_median.loc['male', 1, 'Mr']['Age']
                        elif row['Title'] == 'Officer':
                            return grouped_median.loc['male', 1, 'Officer']['Age']
                        elif row['Title'] == 'Royalty':
                            return grouped_median.loc['male', 1, 'Royalty']['Age']
                        
                    elif row['Sex']=='male' and row['Pclass'] == 2:
                        if row['Title'] == 'Master':
                            return grouped_median.loc['male', 2, 'Master']['Age']
                        elif row['Title'] == 'Mr':
                            return grouped_median.loc['male', 2, 'Mr']['Age']
                        elif row['Title'] == 'Officer':
                            return grouped_median.loc['male', 2, 'Officer']['Age']

                    elif row['Sex']=='male' and row['Pclass'] == 3:
                        if row['Title'] == 'Master':
                            return grouped_median.loc['male', 3, 'Master']['Age']
                        elif row['Title'] == 'Mr':
                            return grouped_median.loc['male', 3, 'Mr']['Age']
    
                self.raw_data.Age = self.raw_data.apply(lambda r : fillAges(r, input_grouped_median) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
            elif col=='Fare':
                # there's one missing fare value - replacing it with the mean.
                self.raw_data.Fare.fillna(self.raw_data.Fare.mean(), inplace=True)
                    
            elif col=='Cabin':
                # replacing missing cabins with U (for Uknown)
                self.raw_data.Cabin.fillna('U', inplace=True)
                
                # mapping each Cabin value with the cabin letter
                self.raw_data['Cabin'] = self.raw_data['Cabin'].map(lambda c : c[0])
                
            else:
                # two missing embarked values - filling them with the most frequent one (S)
                self.raw_data.Embarked.fillna('S', inplace=True)
        
        return self.raw_data      
                
        
        
        
        
        
        
        
        
        
