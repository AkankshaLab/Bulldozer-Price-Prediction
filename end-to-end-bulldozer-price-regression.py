#!/usr/bin/env python
# coding: utf-8

# ### Problem definition
# How well can we predict the sale price of bulldozer, given it features and previous examples of how much similar bulldozers have been sold for?

# ### Data
# Data is downloaded from Kaggle Bluebook for bulldozers competition
# * Train.csv is the training set, which contains data through the end of 2011.
# * Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. 
# * Test.csv is the test set. It contains data from May 1, 2012 - November 2012.

# ### Evaluation
# * The evaluation metric is the RMSLE (root mean squared log error) between the actual and predicted auction prices.
# * The goal for most regression evaluation metrics is to minimize the error.(Minimize RMSLE here)

# ### Features
# Data Dictionary.xlsx

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[4]:


# Import training and validation sets
df = pd.read_csv("TrainAndValid.csv", low_memory = False)
# dataset is large


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.isna().sum()


# In[8]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000]);


# In[9]:


df.SalePrice.plot.hist();


# ###### Parsing dates
# When we work with time series data, we want to enrich the time and date component as much as possible.
# 
# Tell pandas which column has dates using `parse_dates` param

# In[10]:


df.saledate.dtype


# In[11]:


df.saledate[:10]


# In[12]:


df = pd.read_csv("TrainAndValid.csv",
                 low_memory=False,
                 parse_dates=['saledate'])


# In[13]:


df.saledate.dtype


# In[14]:


df.saledate[:10]


# In[15]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000]);


# In[16]:


df.head().T


# In[17]:


df.saledate.head(10)


# In[18]:


# Sort Dataframe by Saledate
df.sort_values(by=["saledate"], 
               inplace=True,
               ascending=True)


# In[19]:


df.saledate.head(10)


# In[20]:


# Make a copy of the original dataframe
# So that when we manipulate the copy, we've still got our original data
df_tmp = df.copy()


# In[21]:


df_tmp.saledate[:10]


# In[22]:


# Add datetime parameters for 'saledate' column


# In[23]:


df_tmp["saleYear"] = df_tmp.saledate.dt.year  # dt stands for datetime
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayOfWeek"] = df_tmp.saledate.dt.day_of_week
df_tmp["saleDayOfYear"] = df_tmp.saledate.dt.day_of_year


# In[24]:


df_tmp[:1].saledate.dt.year


# In[25]:


df_tmp.saledate.dt.year[:1]


# In[26]:


df_tmp[:1].saledate.dt.day


# In[27]:


df_tmp[:1].saledate


# In[28]:


df_tmp.head(200).T


# In[29]:


# We have enriched our dataframe with datetime features, we can remove 'saledate'
df_tmp.drop("saledate", axis=1, inplace=True)


# In[30]:


df_tmp.state.value_counts()


# ### Modelling

# In[31]:


# Model driven EDA


# In[32]:


from sklearn.ensemble import RandomForestRegressor

"""
model = RandomForestRegressor(n_jobs = -1, random_state=42)
model.fit(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])
"""


# In[33]:


# Error coz all data is not in numeric format
# and missing values


# In[34]:


# Convert data in pandas categories


# In[35]:


df_tmp["UsageBand"].dtype


# In[36]:


pd.api.types.is_string_dtype(df_tmp["UsageBand"])


# In[37]:


# Find the columns which contain strings
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)


# In[38]:


# Converting string to categories
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()


# In[39]:


df_tmp.info()


# In[40]:


df_tmp.state.cat.categories


# In[41]:


df_tmp.state.value_counts()


# In[42]:


# Under the hood, pandas has assigned a number to each category
df_tmp.state.cat.codes


# In[43]:


# But we still have missing data


# In[44]:


# Check missing data
df_tmp.isnull().sum()/len(df_tmp)
# tells percentage of missing data in each column


# In[45]:


# Save preprocessed data
# Export current tmp dataframe
df_tmp.to_csv("train_tmp.csv", index=False)


# In[46]:


# Import preprocessed data
df_tmp = pd.read_csv("train_tmp.csv", low_memory=False)


# In[47]:


# Fill missing values


# In[48]:


for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)


# In[49]:


# Check which numeric columns has null value
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[50]:


# Fill numeric rows with median
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells us if the data was missing
            df_tmp[label+"_is_missing"] = pd.isnull(content)
            # fill missing values
            df_tmp[label] = content.fillna(content.median())


# In[51]:


df_tmp.isna().sum()


# In[52]:


# filling and turning categorical variables into numbers


# In[53]:


for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[54]:


pd.Categorical(df_tmp["state"])


# In[55]:


pd.Categorical(df_tmp["state"]).dtype


# In[56]:


pd.Categorical(df_tmp["state"]).codes


# In[57]:


for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample has missing value
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        # Turn categories into numbers and add +1
        df_tmp[label] = pd.Categorical(content).codes + 1 # because missing values have categorical values of -1 


# In[58]:


df_tmp.info()


# In[59]:


df_tmp.head().T


# In[60]:


df_tmp.isna().sum()


# In[61]:


# Modelling


# In[62]:


len(df_tmp)


# In[63]:


get_ipython().run_cell_magic('time', '', '\nmodel = RandomForestRegressor(n_jobs = -1, random_state=42)\nmodel.fit(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])')


# In[64]:


model.score(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])


# In[65]:


# SOlitting data into training and validation test


# In[66]:


df_tmp.saleYear.value_counts()


# In[67]:


df_train = df_tmp[df_tmp.saleYear != 2012]
df_val = df_tmp[df_tmp.saleYear == 2012]


# In[68]:


len(df_train), len(df_val)


# In[69]:


# Split data into X and y
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train["SalePrice"]
X_val, y_val = df_val.drop("SalePrice", axis=1), df_val["SalePrice"]


# In[70]:


X_train.shape, y_train.shape, X_val.shape, y_val.shape


# In[71]:


# Building an evaluation function


# In[72]:


# RMSLE
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_test, y_preds):
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    scores = {"Training MAE" : mean_absolute_error(y_train, train_preds),
              "Validation MAE" : mean_absolute_error(y_val, val_preds),
              "Training RMSLE" : rmsle(y_train, train_preds),
              "Validation RMSLE" : rmsle(y_val, val_preds),
              "Training R^2" : r2_score(y_train, train_preds),
              "Validation R^2" : r2_score(y_val, val_preds)}
    return scores


# In[73]:


# Testing our model on a subset (to tune the hyperparameters)


# In[74]:


"""model = RandomForestRegressor(n_jobs=-1, random_state=42)
model.fit(X_train, y_train)"""
# This takes far too long for experimenting


# In[75]:


# Either train on a fraction of dataset like X_train[:1000]
# Or Change max_samples value
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42,
                              max_samples=10000)


# In[76]:


get_ipython().run_cell_magic('time', '', 'model.fit(X_train, y_train)')


# In[77]:


show_scores(model)


# In[78]:


# Hyperparameter tuning with RandomizedSearchCV


# In[79]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import RandomizedSearchCV\n\nrf_grid = {"n_estimators" : np.arange(10, 100, 10),\n           "max_depth" : [None, 3, 5, 10],\n           "min_samples_split" : np.arange(2, 20, 2),\n           "min_samples_leaf" : np.arange(1, 20, 2),\n           "max_features" : [0.5, 1, "sqrt", "auto"],\n           "max_samples" : [10000]}\n\nrs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1, random_state=42),\n                              param_distributions=rf_grid,\n                              n_iter=2,\n                              cv=5,\n                              verbose=True)\nrs_model.fit(X_train, y_train)')


# In[80]:


rs_model.best_params_


# In[81]:


show_scores(rs_model)


# In[82]:


# Train a model with best hyperparameters
# These were found after n_iter = 100


# In[83]:


get_ipython().run_cell_magic('time', '', '\nideal_model = RandomForestRegressor(n_estimators=40,\n                                    min_samples_leaf=1,\n                                    min_samples_split=14,\n                                    max_features=0.5,\n                                    n_jobs=-1,\n                                    max_samples=None,\n                                    random_state=42)\n\nideal_model.fit(X_train, y_train)')


# In[84]:


show_scores(ideal_model)


# In[85]:


# Import the test data
df_test = pd.read_csv("Test.csv",
                      low_memory=False,
                      parse_dates=["saledate"])
df_test.head()


# In[86]:


# test_preds = ideal_model.predict(df_test)
# Error 
# non numeric data and missing values


# In[87]:


df_test.info()


# In[88]:


df_test.isna().sum()


# In[89]:


len(df_test.columns)


# In[90]:


len(X_train.columns)


# In[91]:


# Preprocessing the data (getting the test dataset in the same format as training dataset)


# In[92]:


def preprocess_data(df):
    
    df["saleYear"] = df.saledate.dt.year 
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.day_of_week
    df["saleDayOfYear"] = df.saledate.dt.day_of_year
    
    df.drop("saledate", axis=1, inplace=True)
    
    # Fill numeric rows with median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing
                df[label+"_is_missing"] = pd.isnull(content)
                # fill missing values
                df[label] = content.fillna(content.median())
            
        # Fill categorical missing data and turn categories into numbers
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            df[label] = pd.Categorical(content).codes+1
    
    return df


# In[93]:


df_test = preprocess_data(df_test)
df_test


# In[94]:


# Make predictions on updated test data
#test_preds = ideal_model.predict(df_test)
#error
X_train.shape, df_test.shape
# still the number of columns is not same


# In[95]:


# Find how columns differ using sets
set(X_train.columns) - set(df_test.columns)


# In[96]:


# all of the auctioneerID were filled in df_test
# so it is not having auctioneerID_is_missing column


# In[97]:


# add auctioneerID_is_missing column
df_test["auctioneerID_is_missing"] = False


# In[98]:


df_test.shape


# In[99]:


# Make predictions on test data
test_preds = ideal_model.predict(df_test)


# In[100]:


test_preds


# In[101]:


len(test_preds)


# In[102]:


# Format predictions as Kaggle is asking


# In[103]:


df_preds = pd.DataFrame()
df_preds["SalesID"] = df_test["SalesID"]
df_preds["SalesPrice"] = test_preds
df_preds


# In[104]:


# Export prediction data 
df_preds.to_csv("test_predictions.csv", index=False)


# #### Feature Importance
# Feature importance seeks to figure out which different attributes of the data were most important when it comes to predicting the target variables (SalePrice)

# In[105]:


# Find feature importance of our best model
ideal_model.feature_importances_


# In[106]:


len(ideal_model.feature_importances_)


# In[107]:


X_train.shape


# In[108]:


X_train.columns


# In[115]:


# Helper finction for plotting feature importance
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features" : columns,
                        "feature_importances" : importances})
         .sort_values("feature_importances", ascending=False)
         .reset_index(drop=True))
    
    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:n])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()


# In[116]:


plot_features(X_train.columns, ideal_model.feature_importances_)


# In[ ]:




