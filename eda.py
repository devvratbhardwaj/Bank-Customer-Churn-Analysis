#%%
## Research Question
## To predict whether a customer retains their account or churns (i.e., closes it)

## Binary classification problem: Whether churned or not

## Evaluation to be done on area under the ROC curve between predicted probability
## and the observed target.

#%%
## Load the datasets and import packages
from turtle import color
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")

#%%
## Exploratory Data Analysis

# train.describe()
# train.head()
# print(train.shape)  # 165034 rows x 14 columns
# train.columns.values
# train.dtypes

## Dependent varaible: Exited (int)

## Independent variables:
    # Categorical variables:
    # Surname, Geography, Gender
    
    # Numerical varaibles: 
    # CustomerId, Credit Score, Age, Tenure, Balance,
    # NumOfProducts, HasCrCard, EstimatedSalary, IsActiveMember                      

## How many people churned
train["Exited"].value_counts()
## -> 34921 customers churned
## OR
train["Exited"].value_counts()/len(train["Exited"])
## i.e., 21.1% customers churned. 
## i.e., imbalanced data. Can lead to overfitting.
## Overfitting: Predictive models give high accuracy for training
## data, but gives poor accuracy for test data. i.e., accuracy is cursed for imbalanced dataset 
# aka Biased Model
# We need to balance the dataset to create an unbalanced dataset.

## Upsampling
# Creating synthetic records for the lagging class
# Gives better results (in 80% cases) as there is more data
# Therefore it is recommended. But do hit and trail with downsampling as well.

## Downsampling
# Arbitraility remove records from the excess class to balance
# it with the lagging class
# The loss of data might also remove useful information. 

#%%
## Do we have any missing data
train.info()
## There are no missing values (all non-null)
## No need to drop any columns or fabricate values

## If there is a very-low percentage of missing values
## ignore them by dropping the records (dropna)

## If there is a low percentage of missing values (30%)
## we can use regression or use mean to fill the missing values

## If there is a high percentage of missing values
## then it is better to drop the column

## In some scenarios we have to convert
## object to numerical values to see whether they are null
## or not.

#%%
## Irrelevant columns would be: Surname, CustomerID, id
## Drop these columns

train.drop(columns=["Surname", "CustomerId", "id"], axis=1,inplace=True)
#%%
## max and min tenure
min_tenure = train["Tenure"].min()
max_tenure = train["Tenure"].max()

## Form bins of width 2 years
bins = [0,2,4,6,8,10]
train["tenure_bins"] = pd.cut(train["Tenure"],bins, include_lowest=True)
# print(train)

train["tenure_bins"].value_counts()
# print(train["tenure_bins"].value_counts().sum())

## drop tenure as well
train.drop(columns=["Tenure"], axis=1,inplace=True)
train.head()

#%%
## Rough analysis of categorical variables

## Churners according to Gender
sns.countplot(data = train, x = "Gender", hue="Exited")
## Ratio of churners to non-churners is higher in females

#%%
## Churners according to Geography
sns.countplot(data = train, x = "Geography", hue="Exited")
## France has the highest churners followed by Germany
## Spain has the lowest churners

#%%
## Convert categorical variables into numerical variables
## by One-hot encoding
## Label-encoding is reserved for target variables

## One-Hot encoding Gender and dummy trapping
train_dummies = pd.get_dummies(train, dtype='int')
train_dummies.head()
train_dummies.shape
## One-Hot encoding Geography and dummy trapping

#%%
## The correlation-heatmap
plt.figure(figsize= (12,12))
cmap = sns.diverging_palette(130, 80, as_cmap=True)
sns.heatmap(train_dummies.corr(), cmap= cmap)

#%%
train_dummies.to_csv("clean.csv")


#%%
## Process the test data as done with the training data
test_file = pd.read_csv("test.csv")
test_file.drop(columns=["Surname", "CustomerId"], axis=1,inplace=True)

## Bin the tenure
bins = [0,2,4,6,8,10]
test_file["tenure_bins"] = pd.cut(test_file["Tenure"],bins, include_lowest=True)
# print(train)
test_file["tenure_bins"].value_counts()
# print(train["tenure_bins"].value_counts().sum())

## drop tenure as well
test_file.drop(columns=["Tenure"], axis=1,inplace=True)

## One hot encode
test_dummy =  pd.get_dummies(test_file, dtype='int')
# test_dummy.head()
test_dummy.to_csv("clean_test.csv")