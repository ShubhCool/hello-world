# -*- coding: utf-8 -*-
"""
Created on Sun May 13 19:57:21 2018

@author: shubham
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from  scipy  import stats

%matplotlib 



credit_df = pd.read_csv( 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',delimiter=' ',header=None)

#credit_df.head()

columns = ['checkin_acc', 'duration', 'credit_history', 'purpose', 'amount',
         'saving_acc', 'present_emp_since', 'inst_rate', 'personal_status',
         'other_debtors', 'residing_since', 'property', 'age',
         'inst_plans', 'housing', 'num_credits',
         'job', 'dependents', 'telephone', 'foreign_worker', 'status']

credit_df.columns = columns

#credit_df.head()

#credit_df.info()

credit_df.status = credit_df.status - 1

#distribution of credit amount
sn.distplot( credit_df.amount, kde = False )
plt.title( "Histogram of Credit Amount Disbursed", fontsize = 15)
plt.ylabel( "Frequency")
#Most of the credit amounts are less than 5000 with some higher credit amounts. The largest amount disbursed is as high as 18000+
#credit_df.amount.describe()

sn.boxplot( credit_df.amount, orient = 'v' )
plt.title( "Boxplot of Credit Amount Disbursed", fontsize = 15)
#The middle 50% of the population lies between 1300 to 3900

sn.boxplot( x = 'status', y = 'amount', data = credit_df, orient = 'v' )
plt.title( "Boxplot of Credit Amount Disbursed by Credit Status", fontsize = 15)
#Lot of higher credit amounts seem to have been defaulted


#Distribution plot for credit amounts for differernt status
sn.distplot( credit_df[credit_df.status == 0].amount, color = 'g', hist = False )
sn.distplot( credit_df[credit_df.status == 1].amount, color = 'r', hist = False )
plt.title( "Distribution plot of Amount comparison for Different Credit Status", fontsize = 10 )
plt.ylabel( "Frequency")

g = sn.FacetGrid(credit_df, col="status", size = 6)
g.map(sn.distplot, "amount", kde = False, bins = 20 )
#Amounts higher than 10000 have been mostly defaulted


#credit default rate
d_rate_df = pd.DataFrame( credit_df.status.value_counts( normalize=True ) )
#d_rate_df

sn.barplot( x = d_rate_df.index, y = d_rate_df.status )
#credit_df.amount.describe()

amount_desc = credit_df.amount.describe()
outliers = amount_desc['75%'] + 1.5 * ( amount_desc['75%'] - amount_desc['25%'] )
#outliers

d_rate_outliers_df = pd.DataFrame( credit_df[credit_df.amount >
                      outliers ].status.
                   value_counts( normalize = True ) )
sn.barplot( x = d_rate_outliers_df.index, y = d_rate_outliers_df.status )

extreme_outliers = amount_desc['75%'] + 3 * ( amount_desc['75%'] - amount_desc['25%'] )
extreme_outliers_df = pd.DataFrame( credit_df[credit_df.amount >
                      extreme_outliers ].status.
                    value_counts( normalize = True ) )
sn.barplot( x = extreme_outliers_df.index, y = extreme_outliers_df.status )


#Analyzing Interest Rates on Credit Defaults
credit_df.inst_rate.unique()

rate_count = credit_df[['inst_rate', 'status']].groupby(['inst_rate', 'status']).size().reset_index()
rate_count.columns = ['inst_rate', 'status', 'count']

#rate_count

g = sn.factorplot(x="inst_rate", y = 'count', hue="status", data=rate_count,
                 size=6, kind="bar", palette="muted")

#Credit Amount for different interest rates and their impact on bad credit
sn.boxplot( x = 'inst_rate', y = 'amount', hue = 'status', data = credit_df, orient = 'v' )
plt.title( "Boxplot of Credit Amount Disbursed by Credit Status", fontsize = 12)

sn.lmplot( x = 'inst_rate', y = 'amount', data = credit_df )

# average credit amount for interest rate = 1 different for good and bad credit 

credit_inst_rate_1_df = credit_df[ credit_df.inst_rate == 1 ]
sn.distplot( credit_inst_rate_1_df[credit_inst_rate_1_df.status == 0 ].amount, color = 'g', hist = False )
sn.distplot( credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1].amount, color = 'r', hist = False )
sn.plt.axvline( x = credit_inst_rate_1_df[credit_inst_rate_1_df.status == 0 ].amount.mean(), color = 'g' )
sn.plt.axvline( x = credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1 ].amount.mean(), color = 'r' )
plt.title( "Distribution plot of Amount Disbured for inst_rate = 1", fontsize = 10 )
plt.ylabel( "Frequency")


'''For interest rate = 1, the average credit amount for bad credits seems to be higher then good credits
Hypothesis Test
H0 : average Credit amount for good credit = average credit amount for bad credit

H1 : average Credit amount for good credit <> average credit amount for bad credit'''

stats.ttest_ind( credit_inst_rate_1_df[credit_inst_rate_1_df.status == 0 ].amount,
               credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1 ].amount)
'''p-value < 0.05 indicates average credit for good credit is less than average credit amount for bad credits
Average credit amount for bad credit'''

credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1 ].amount.mean()

credit_inst_rate_1_df[credit_inst_rate_1_df.status == 0 ].amount.mean()

'''cutoff for rejecting credit amount for inst_rate = 1? What is the reduction in default and what is opportunity cost'''
stats.ttest_1samp( credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1 ].amount, 4700 )

1 - stats.norm.cdf( 4700,
                  loc = credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1 ].amount.mean() ,
                  scale = credit_inst_rate_1_df[credit_inst_rate_1_df.status == 1 ].amount.bstd() )

credit_inst_rate_1_df[ (credit_inst_rate_1_df.status == 1)
                    & (credit_inst_rate_1_df.amount > 4700)].amount.sum()

1 - stats.norm.cdf( 4700,
                  loc = credit_inst_rate_1_df[credit_inst_rate_1_df.status == 0 ].amount.mean() ,
                  scale = credit_inst_rate_1_df[credit_inst_rate_1_df.status == 0 ].amount.std() )

credit_inst_rate_1_df[ (credit_inst_rate_1_df.status == 0)
                    & (credit_inst_rate_1_df.amount > 4700)].amount.sum()
'''Assuming there is 10% return on the credits, the bank is loosing only 16636.6, where as the loss becuase of bad credit is complete principal amount i.e. 177609, which is very high'''


#impact of customer having checkin account on bad credits¶

sn.barplot( x = 'checkin_acc', y = 'amount', hue = 'status', data = credit_df )
plt.title( "Average credit amount by different checkin account holders")
plt.figtext(1, 0.5,"""A11 : < 0 DM \n A12 : 0 <= ... < 200 DM \n A13 : >= 200 DM \n A14 : no checking account 
""", wrap=True, horizontalalignment='left', fontsize=12)

sn.barplot( x = 'checkin_acc',
         y = 'amount',
         hue = 'status',
         data = credit_df,
         estimator = sum )
plt.title( "Total credit amount by different checkin account holders")
plt.figtext(1, 0.5,"""A11 : < 0 DM \n A12 : 0 <= ... < 200 DM \n A13 : >= 200 DM \n A14 : no checking account 
""", wrap=True, horizontalalignment='left', fontsize=12)

sn.countplot( y = 'checkin_acc', hue = 'status', data = credit_df )

#Customers having no checkin account seems to have lesser chance of making a default where as customers having checkin accoun without any balance have higher chance of making a default

#Impact of credit history on bad credits
figure_text = """A30 : no credits taken/ all credits paid back duly \n
A31 : all credits at this bank paid back duly \n
A32 : existing credits paid back duly till now \n
A33 : delay in paying off in the past \n
A34 : critical account/ other credits existing (not at this bank) """

sn.barplot( x = 'credit_history', y = 'amount', hue = 'status', data = credit_df, estimator = sum )
plt.figtext(1, 0.5,figure_text, wrap=True, horizontalalignment='left', fontsize=12)

sn.countplot( y = 'credit_history', hue = 'status', data = credit_df )
plt.figtext(1, 0.5,figure_text, wrap=True, horizontalalignment='left', fontsize=12)

#Analyzing impact of credit purpose on bad credit

purpose_text = '''
A40 : car (new) \n
A41 : car (used) \n
A42 : furniture/equipment \n
A43 : radio/television \n
A44 : domestic appliances \n 
A45 : repairs \n
A46 : education \n
A47 : (vacation - does not exist?) \n
A48 : retraining \n
A49 : business \n
A410 : others '''


snsn..barplotbarplot((  xx  ==  'purpose''purpose', y = 'amount', hue = 'status', data = credit_df )
plt.figtext(1, 0.3,purpose_text, wrap=True, horizontalalignment='left', fontsize=8)

sn.barplot( x = 'purpose', y = 'amount', hue = 'status', data = credit_df, estimator = sum )
plt.figtext(1, 0.3,purpose_text, wrap=True, horizontalalignment='left', fontsize=8)

'''average loan amount taken for used car purchse differnent for defaulters and non-defaulters'''

credit_used_car_df = credit_df[ credit_df.purpose == 'A41' ]
sn.distplot( credit_used_car_df[credit_used_car_df.status == 0 ].amount, color = 'g', hist = False )
sn.distplot( credit_used_car_df[credit_used_car_df.status == 1].amount, color = 'r', hist = False )
sn.plt.axvline( x = credit_used_car_df[credit_used_car_df.status == 0 ].amount.mean(), color = 'g' )
sn.plt.axvline( x = credit_used_car_df[credit_used_car_df.status == 1 ].amount.mean(), color = 'r' )
plt.title( "Distribution plot of Amount Disbured for Used Car Purchase and Credit Status", fontsize = 10 )
plt.ylabel( "Frequency")

stats.ttest_ind( credit_used_car_df[credit_used_car_df.status == 0 ].amount,
               credit_used_car_df[credit_used_car_df.status == 1 ].amount)

#average loan amount taken for new car purchse differnent for defaulters and non-defaulters

credit_new_car_df = credit_df[ credit_df.purpose == 'A40' ]
sn.distplot( credit_new_car_df[credit_new_car_df.status == 0 ].amount, color = 'g', hist = False )
sn.distplot( credit_new_car_df[credit_new_car_df.status == 1].amount, color = 'r', hist = False )
sn.plt.axvline( x = credit_new_car_df[credit_new_car_df.status == 0 ].amount.mean(), color = 'g' )
sn.plt.axvline( x = credit_new_car_df[credit_new_car_df.status == 1 ].amount.mean(), color = 'r' )
plt.title( "Distribution plot of Amount Disbured for Used Car Purchase and Credit Status", fontsize = 10 )
plt.ylabel( "Frequency")

stats.ttest_ind( credit_new_car_df[credit_new_car_df.status == 0 ].amount,
               credit_new_car_df[credit_new_car_df.status == 1 ].amount)

#average loan amount taken for domestic appliances purchse differnent for defaulters and non-defaulters
credit_appliances_df = credit_df[ credit_df.purpose == 'A44' ]
sn.distplot( credit_appliances_df[credit_appliances_df.status == 0 ].amount, color = 'g', hist = False )
sn.distplot( credit_appliances_df[credit_appliances_df.status == 1].amount, color = 'r', hist = False )
sn.plt.axvline( x = credit_appliances_df[credit_appliances_df.status == 0 ].amount.mean(), color = 'g' )
sn.plt.axvline( x = credit_appliances_df[credit_appliances_df.status == 1 ].amount.mean(), color = 'r' )
plt.title( "Distribution plot of Amount Disbured for Used Car Purchase and Credit Status", fontsize = 10 )
plt.ylabel( "Frequency")

stats.ttest_ind( credit_appliances_df[credit_appliances_df.status == 0 ].amount,
               credit_appliances_df[credit_appliances_df.status == 1 ].amount)

sn.countplot( x = 'purpose', hue = 'status', data = credit_df )
plt.figtext(1, 0.3,purpose_text, wrap=True, horizontalalignment='left', fontsize=8)

#Relationship betweeb Credit amount and Duration
sn.lmplot( x = 'duration', y = 'amount', fit_reg = False, data = credit_df )

sn.lmplot( x = 'duration', y = 'amount', fit_reg = True, data = credit_df )


sn.lmplot(  x =  'duration', y = 'amount', hue = 'status', fit_reg = False, data = credit_df )

#Customers taking credit for large amount for lesser duration are making default mostly. This kind of loans can be restricted


#Impact of age on bad credits
sn.lmplot( x = 'age', y = 'amount', hue = 'status', fit_reg = True, data = credit_df )

#Understanding relationship between continuous variables using pair plot

credit_df.select_dtypes(include = ['float64', 'int64'])[0:5]
sn.pairplot( credit_df.select_dtypes(include = ['float64', 'int64']).iloc[:, :-1] )

#Impact of personal status on bad credit


personal_textpersonal_  = '''A91 : male : divorced/separated \n
A92 : female : divorced/separated/married \n
A93 : male : single \n
A94 : male : married/widowed \n
A95 : female : single'''

sn.barplot( x = 'personal_status', y = 'amount', hue = 'status', data = credit_df, estimator = sum )
plt.figtext(1, 0.3,personal_text, wrap=True, horizontalalignment='left', fontsize=8)

sn.barplot( x = 'personal_status', y = 'amount', hue = 'status', data = credit_df )
plt.figtext(1, 0.3,personal_text, wrap=True, horizontalalignment='left', fontsize=8)

credit_men_df = credit_df[ credit_df.personal_status == 'A93' ]
sn.distplot( credit_men_df[credit_men_df.status == 0 ].amount, color = 'g', hist = False )
sn.distplot( credit_men_df[credit_men_df.status == 1].amount, color = 'r', hist = False )
sn.plt.axvline( x = credit_men_df[credit_men_df.status == 0 ].amount.mean(), color = 'g' )
sn.plt.axvline( x = credit_men_df[credit_men_df.status == 1 ].amount.mean(), color = 'r' )
plt.title( "Distribution plot of Amount comparison for Different Credit Status", fontsize = 10 )
plt.ylabel( "Frequency")

stats.ttest_ind( credit_men_df[credit_men_df.status == 0 ].amount,
               credit_men_df[credit_men_df.status == 1 ].amount,
               equal_var = False )

#Impact of number of dependents on bad credits

sn.countplot( x = 'personal_status', hue = 'status', data = credit_df )
plt.figtext(1, 0.3,personal_text, wrap=True, horizontalalignment='left', fontsize=8)

pd.crosstab( credit_df.dependents, credit_df.status )

credit_df.dependents.unique()

sn.barplot( x = 'dependents', y = 'amount', hue = 'status', data = credit_df )

'''Analyze impact of saving_account, present_employment_since, property, foreign_worker on good and bad_credit and conclude some strategies for avoiding bad credits.'''














































