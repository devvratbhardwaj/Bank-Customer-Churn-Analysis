#%%
## Building models
from typing import final
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

df = pd.read_csv("clean.csv")
df = df.drop("Unnamed: 0", axis=1)
df.head()

X = df.drop("Exited", axis=1)
# X.columns
Y = df["Exited"]

## Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)  ## 20% of data is test data

# #%%
# ## Using decision tree classifier

# model = DecisionTreeClassifier(criterion='gini', max_depth=7, random_state=100) 
# ## Can tune/optimize the hyperparameters later on using Grid Search or Bayesian Optimization
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# print(classification_report(y_test, y_pred, labels=[0,1]))
# ## Recall value doesn't look good for churners
# ## Precision is fine
# # print(confusion_matrix(y_test,y_pred))

#%% Re-sampling
from imblearn.combine import SMOTEENN

sm = SMOTEENN()
x_resampled, y_resampled = sm.fit_resample(X,Y)
xre_train, xre_test, yre_train, yre_test = train_test_split(x_resampled,y_resampled,test_size=0.2)  ## 20% of data is test data

#%%

# ## Decision Tree Classifier
# model = DecisionTreeClassifier(criterion='gini', max_depth=7, random_state=100) 
# model.fit(xre_train, yre_train)

# yre_pred = model.predict(xre_test)

# print(classification_report(yre_test, yre_pred, labels=[0,1]))
# ## Now the model is good

#%%
## Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators = 70, criterion = 'gini', max_depth = 11, random_state = 100)
rf_model.fit(xre_train, yre_train)
yre_pred = rf_model.predict(xre_test)

print(classification_report(yre_test, yre_pred, labels=[0,1]))
## Fine-tuned model gives 93% precision and 93% recall for churners

#%%
test_data = pd.read_csv("clean_test.csv")
test_data = test_data.drop("Unnamed: 0", axis =1)
test_data.head()
#%%
x_new_test = test_data.iloc[:,1:]
x_new_test.head()

pred = rf_model.predict_proba(x_new_test)
print(pred)
#%%
submission = test_data["id"].to_frame()

column_values = pd.Series(pred[:,1])

submission.insert(loc=1, column= "Exited", value=column_values)

submission.to_csv("submission.csv", index=False)
#%%
## Saving the models using pickle