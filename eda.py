#%%
## Research Question
## To predict whether a customer retains their account or churns (i.e., closes it)

## Binary classification problem: Whether churned or not

## Evaluation to be done on area under the ROC curve between predicted probability
## and the observed target.

#%%
## Load the datasets and import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

training_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#%%
## Exploratory Data Analysis

# training_data.describe()
# training_data.head()
# print(training_data.shape)  # 165034 rows x 14 columns
# training_data.columns.values
# training_data.dtypes

## Dependent varaible: Exited (int)

## Independent variables:
    # Categorical variables:
    # Surname, Geography, Gender
    
    # Numerical varaibles: 
    # CustomerId, Credit Score, Age, Tenure, Balance,
    # NumOfProducts, HasCrCard, EstimatedSalary, IsActiveMember                      

## How many people churned
training_data["Exited"].value_counts()
## -> 34921 customers churned
## OR
training_data["Exited"].value_counts()/len(training_data["Exited"])
## i.e., 21.1% customers churned. 
## i.e., imbalanced data. Can lead to overfitting.
## Overfitting: Predictive models performs well for training
## data, but performs poorly for test data.

