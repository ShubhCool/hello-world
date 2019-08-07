#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:44:03 2019

@author: shubham
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler

class Feature_Engineering:
    """ 
    This is a class for Feature engineering in order to derive new feature
    and also help with imputation. 
      
    Attributes: 
         
    """
    scaler=None
        
    def derive_new_cols(self,combined_data:pd.DataFrame) -> pd.DataFrame:
        '''
        The function to list out the different variable types. 
  
        Parameters: 
            self: class object
        Returns: 
            None
        '''
        # we extract the title from each name
        combined_data['Title'] = combined_data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
        # a map of more aggregated titles
        Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
        # we map each title
        combined_data['Title'] = combined_data.Title.map(Title_Dictionary)
        
        # introducing a new feature : the size of families (including the passenger)
        combined_data['FamilySize'] = combined_data['Parch']
        + combined_data['SibSp'] + 1
    
        # introducing other features based on the family size
        combined_data['Singleton'] = combined_data['FamilySize'].map(lambda s: 
            1 if s == 1 else 0)
        combined_data['SmallFamily'] = combined_data['FamilySize'].map(lambda s: 
            1 if 2<=s<=4 else 0)
        combined_data['LargeFamily'] = combined_data['FamilySize'].map(lambda s: 
            1 if 5<=s else 0)
            
        return combined_data
    
    def dummy_encoding(self,input_data:pd.DataFrame,col_list:list)->pd.DataFrame:
        
        for col in col_list:
            
            if col=='Sex':
                # mapping string values to numerical one 
                input_data['Sex'] = input_data['Sex'].map({'male':1,'female':0})
            
            elif col=='Pclass':
                # encoding into 3 categories:
                pclass_dummies = pd.get_dummies(input_data['Pclass'], prefix="Pclass")
                # adding dummy variables
                input_data = pd.concat([input_data,pclass_dummies],axis=1)
                # removing "Pclass"
                input_data.drop('Pclass',axis=1,inplace=True)
                
            elif col=='Embarked':
                # dummy encoding 
                embarked_dummies = pd.get_dummies(input_data['Embarked'],prefix='Embarked')
                input_data = pd.concat([input_data,embarked_dummies],axis=1)
                input_data.drop('Embarked',axis=1,inplace=True)
                
            elif col=='Cabin':
                # dummy encoding ...
                cabin_dummies = pd.get_dummies(input_data['Cabin'], prefix='Cabin') 
                input_data = pd.concat([input_data,cabin_dummies], axis=1) 
                input_data.drop('Cabin', axis=1, inplace=True)
                
            else:
                # we clean the Name variable
                input_data.drop('Name',axis=1,inplace=True)
                # encoding in dummy variable
                titles_dummies = pd.get_dummies(input_data['Title'],prefix='Title')
                input_data = pd.concat([input_data,titles_dummies],axis=1)
                # removing the title variable
                input_data.drop('Title',axis=1,inplace=True)
                #we also drop Ticket variable
                input_data.drop('Ticket',axis=1,inplace=True)
                #we also drop Passenger ID
                input_data.drop('PassengerId',axis=1,inplace=True)
                
        return input_data 

    def input_feat_scale(self,input_data:pd.DataFrame,train_yes:str)->pd.DataFrame:
        if train_yes==True:
            Feature_Engineering.scaler=StandardScaler()
            input_data=Feature_Engineering.scaler.fit_transform(input_data)

        else:
            input_data=Feature_Engineering.scaler.transform(input_data)
        
        return input_data
            
        

    

