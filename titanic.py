# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:46:14 2018

@author: shubham
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


filepath='C:\\Users\\shubham\\Desktop\\implement\\train.csv'
data=pd.read_csv(filepath,index_col='PassengerId')


# data cleaning steps
# check value count and info of columns 
data.info()
# use df.any() df.isnull().any() df.notnull().all() to check for missing values
# replacing age column nan values with mean of columns

data['Age']=data['Age'].fillna(np.mean(data['Age']))

#drop insignificant columns
data=data.drop(['Name','Ticket','Cabin'],axis=1)

# deleting embarked column missing entries
data=data.dropna()

# data visualisation  
# try histogram and boxplot
#data.Pclass.plot('hist')
#data.plot(kind='box',x='Survived',y='Pclass')

#plt.show()

# data type conversion
data['Pclass']=data['Pclass'].astype('category')
data['Sex']=data['Sex'].astype('category')
#data['SibSp']=data['SibSp'].astype('category')
#data['Parch']=data['Parch'].astype('category')
data['Embarked']=data['Embarked'].astype('category')

# conversion from category to dummy variables
data=pd.get_dummies(data)

# removal of duplicated column
data=data.drop(['Pclass_1','Sex_female','Embarked_C'],axis=1)


#seperating X and Y
x=data.iloc[:,1:].values
y=(data.iloc[:,0]).values

# scaling
from sklearn.preprocessing import scale
x=scale(x)

# seperating into train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

# using logistic Regression 
from sklearn.linear_model import LogisticRegression
LogReg=LogisticRegression(random_state=0)
LogReg.fit(x_train,y_train)
y_pred=LogReg.predict(x_test)

#checking performances
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print(accuracy_score(y_test,y_pred))
print(LogReg.score(x_test,y_test))
print(confusion_matrix(y_test,y_pred))



from sklearn.model_selection import cross_val_score
c,r=y_train.shape
y_train=y_train.reshape(c,)
cv_result=cross_val_score(LogReg,x_train,y_train,cv=5)



# taking test data as different dataset
filepath1='C:\\Users\\shubham\\Desktop\\implement\\test.csv'
data1=pd.read_csv(filepath1,index_col='PassengerId')

# data cleaning steps
# check value count and info of columns 
data1.info()
# use df.any() df.isnull().any() df.notnull().all() to check for missing values
# replacing age column nan values with mean of columns

data1['Age']=data1['Age'].fillna(np.mean(data1['Age']))
data1['Fare']=data1['Fare'].fillna(np.mean(data1['Fare']))

#drop insignificant columns
data1=data1.drop(['Name','Ticket','Cabin'],axis=1)


# data visualisation  
# try histogram and boxplot
#data.Pclass.plot('hist')
#data.plot(kind='box',x='Survived',y='Pclass')

#plt.show()

# data type conversion
data1['Pclass']=data1['Pclass'].astype('category')
data1['Sex']=data1['Sex'].astype('category')
#data1['SibSp']=data1['SibSp'].astype('category')
#data1['Parch']=data1['Parch'].astype('category')
data1['Embarked']=data1['Embarked'].astype('category')

# conversion from category to dummy variables
data1=pd.get_dummies(data1)

# removal of duplicated column
data1=data1.drop(['Pclass_1','Sex_female','Embarked_C'],axis=1)


#seperating X and Y
x_test1=data1.values
y_pred1=LogReg.predict(x_test1)


# transforming final result into submittable form
data3=pd.DataFrame({'PassengerId':data1.index,'Survived':y_pred1},index=None)
data3.to_csv('C:\\Users\\shubham\\Desktop\\implement\\pred.csv')


