#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:44:27 2019

@author: shubham
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle

class Model_manager:
    """ 
    This is a class for Feature engineering in order to derive new feature
    and also help with imputation. 
      
    Attributes: 
         
    """        
    def Tree_Based_FS(self,train,targets)->bytes:
        
        clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
        clf = clf.fit(train, targets)

        features = pd.DataFrame()
        features['feature'] = train.columns
        features['importance'] = clf.feature_importances_
        features.sort_values(by=['importance'], ascending=True, inplace=True)
        features.set_index('feature', inplace=True)

        #features.plot(kind='bar', figsize=(10, 5))
        
        model = SelectFromModel( clf, prefit=True,threshold=0.01)

        
        return pickle.dumps(model)
    
    def gbm_model_train(self,train,targets,run_gs):
        
        if run_gs==False:
            gbm0 = GradientBoostingClassifier(random_state=0)
            gbm0.fit(train,targets)
            cv_result=cross_val_score(gbm0,train,targets,cv=5)
            print('gradient boosting cross validation score is ',cv_result.mean() )
            
        else:
            #using grid search CV with random forest  classfier
            rf=RandomForestClassifier(random_state=0)
            parameters = {
                 'max_depth' : [6, 8,10],
                 'n_estimators': [50, 100,200,400],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [3,5, 10],
                 'min_samples_leaf': [5, 10, 15],
                 'bootstrap': [True, False],
                 'criterion':['gini','entropy']
                 }
            grid_sear=GridSearchCV(rf,param_grid=parameters,scoring='accuracy',cv=10)
            grid=grid_sear.fit(train,targets)
            print(grid.best_score_)
            print(grid.best_params_)
            
            
    def rf_model_train(self,train,targets,run_gs)->bytes:
        
        if run_gs==False:
            rf=RandomForestClassifier(bootstrap=True, criterion= 'entropy', max_depth=10, max_features= 'sqrt', min_samples_leaf= 10, min_samples_split= 3, n_estimators= 50)
            rf.fit(train,targets)
            cv_result=cross_val_score(rf,train,targets,cv=5)
            print('random forest cross validation score is ',cv_result.mean() )
            return pickle.dumps(rf)
            
        else:
            #using grid search CV with random forest  classfier
            rf=RandomForestClassifier(random_state=0)
            parameters = {
                 'max_depth' : [6, 8,10],
                 'n_estimators': [50, 100,200,400],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [3,5, 10],
                 'min_samples_leaf': [5, 10, 15],
                 'bootstrap': [True, False],
                 'criterion':['gini','entropy']
                 }
            grid_sear=GridSearchCV(rf,param_grid=parameters,scoring='accuracy',cv=10)
            grid=grid_sear.fit(train,targets)
            print(grid.best_score_)
            print(grid.best_params_)       
        return pickle.dumps(grid)
        
        
        
        
        
        
        
        
        
        
        
        
        
        