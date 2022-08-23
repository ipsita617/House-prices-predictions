#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction with Linear Regression
# 
# ![](https://i.imgur.com/3sw1fY9.jpg)
# 
# In this assignment, I am going to predict the price of a house using information like its location, area, no. of rooms etc. You'll use the dataset from the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition on [Kaggle](https://kaggle.com). The step-by-step process to train the model:
# 
# 1. Download and explore the data
# 2. Prepare the dataset for training
# 3. Train a linear regression model
# 4. Make predictions and evaluate the model
# 

# In[1]:


get_ipython().system('pip install jovian scikit-learn --upgrade --quiet')


# In[2]:


import jovian


# In[3]:


jovian.commit(project='python-sklearn-assignment', privacy='secret')


# Let's begin by installing the required libraries:

# In[4]:


get_ipython().system('pip install numpy pandas matplotlib seaborn plotly opendatasets jovian --quiet')


# ## Step 1 - Download and Explore the Data
# 
# The dataset is available as a ZIP file at the following url:

# In[5]:


dataset_url = 'https://github.com/JovianML/opendatasets/raw/master/data/house-prices-advanced-regression-techniques.zip'


# We'll use the `urlretrieve` function from the module [`urllib.request`](https://docs.python.org/3/library/urllib.request.html) to dowload the dataset.

# In[6]:


from urllib.request import urlretrieve


# In[7]:


urlretrieve(dataset_url, 'house-prices.zip')


# The file `housing-prices.zip` has been downloaded. Let's unzip it using the [`zipfile`](https://docs.python.org/3/library/zipfile.html) module.

# In[8]:


from zipfile import ZipFile


# In[9]:


with ZipFile('house-prices.zip') as f:
    f.extractall(path='house-prices')


# The dataset is extracted to the folder `house-prices`. Let's view the contents of the folder using the [`os`](https://docs.python.org/3/library/os.html) module.

# In[10]:


import os


# In[11]:


data_dir = 'house-prices'


# In[12]:


os.listdir(data_dir)


# Use the "File" > "Open" menu option to browse the contents of each file. You can also check out the [dataset description](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) on Kaggle to learn more.
# 
# We'll use the data in the file `train.csv` for training our model. We can load the for processing using the [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html) library.

# In[13]:


import pandas as pd
pd.options.display.max_columns = 200
pd.options.display.max_rows = 200


# In[14]:


train_csv_path = data_dir + '/train.csv'
train_csv_path


# > **QUESTION 1**: Load the data from the file `train.csv` into a Pandas data frame.

# In[15]:


prices_df = pd.read_csv(train_csv_path)


# In[16]:


prices_df


# Let's explore the columns and data types within the dataset.

# In[17]:


prices_df.info()


# > **QUESTION 2**: How many rows and columns does the dataset contain? 

# In[18]:


n_rows = len(prices_df.axes[0])


# In[19]:


n_cols = len(prices_df.axes[1])


# In[20]:


print('The dataset contains {} rows and {} columns.'.format(n_rows, n_cols))


# ## Visualization

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', None)


# In[22]:


px.histogram(prices_df, 
             x='SalePrice', 
             y = 'MasVnrArea',
             title='MasVnrArea vs. SalePrice')


# Let's save our work before continuing.

# In[23]:


px.scatter(prices_df, 
           title='LotFrontage & MSZoning Relationship with SalePrice',
           x='LotFrontage', 
           y='SalePrice',
           color = 'MSZoning')


# ## Step 2 - Prepare the Dataset for Training
# 
# Before we can train the model, we need to prepare the dataset. Here are the steps we'll follow:
# 
# 1. Identify the input and target column(s) for training the model.
# 2. Identify numeric and categorical input columns.
# 3. [Impute](https://scikit-learn.org/stable/modules/impute.html) (fill) missing values in numeric columns
# 4. [Scale](https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range) values in numeric columns to a $(0,1)$ range.
# 5. [Encode](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features) categorical data into one-hot vectors.
# 6. Split the dataset into training and validation sets.
# 

# ### Identify Inputs and Targets
# 
# While the dataset contains 81 columns, not all of them are useful for modeling. Note the following:
# 
# - The first column `Id` is a unique ID for each house and isn't useful for training the model.
# - The last column `SalePrice` contains the value we need to predict i.e. it's the target column.
# - Data from all the other columns (except the first and the last column) can be used as inputs to the model.
#  

# In[24]:


prices_df


# In[25]:


# Identify the input columns (a list of column names)
input_cols =  list(prices_df.columns)[1:-1]


# In[26]:


# Identify the name of the target column (a single string, not a list)
target_col = 'SalePrice'


# In[27]:


print(list(input_cols))


# In[28]:


len(input_cols)


# In[29]:


print(target_col)


# Make sure that the `Id` and `SalePrice` columns are not included in `input_cols`.
# 
# Now that we've identified the input and target columns, we can separate input & target data.

# In[30]:


inputs_df = prices_df[input_cols].copy()


# In[31]:


targets = prices_df[target_col]


# In[32]:


inputs_df


# In[33]:


targets


# ### Identify Numeric and Categorical Data
# 
# The next step in data preparation is to identify numeric and categorical columns. We can do this by looking at the data type of each column.

# In[34]:


prices_df.info()


# > **QUESTION 4**: Crate two lists `numeric_cols` and `categorical_cols` containing names of numeric and categorical input columns within the dataframe respectively. Numeric columns have data types `int64` and `float64`, whereas categorical columns have the data type `object`.
# >
# > *Hint*: See this [StackOverflow question](https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas). 

# In[35]:


import numpy as np


# In[36]:


numeric_cols = inputs_df.select_dtypes(include=['int64', 'float64']).columns.tolist()


# In[37]:


categorical_cols =inputs_df.select_dtypes(include=['object']).columns.tolist()


# In[38]:


print(list(numeric_cols))


# In[39]:


print(list(categorical_cols))


# ### Impute Numerical Data
# 
# Some of the numeric columns in our dataset contain missing values (`nan`).

# In[40]:


missing_counts = inputs_df[numeric_cols].isna().sum().sort_values(ascending=False)
missing_counts[missing_counts > 0]


# Machine learning models can't work with missing data. The process of filling missing values is called [imputation](https://scikit-learn.org/stable/modules/impute.html).
# 
# <img src="https://i.imgur.com/W7cfyOp.png" width="480">
# 
# There are several techniques for imputation, but we'll use the most basic one: replacing missing values with the average value in the column using the `SimpleImputer` class from `sklearn.impute`.
# 

# In[41]:


from sklearn.impute import SimpleImputer


# > **QUESTION 5**: Impute (fill) missing values in the numeric columns of `inputs_df` using a `SimpleImputer`. 
# >
# > *Hint*: See [this notebook](https://jovian.ai/aakashns/python-sklearn-logistic-regression/v/66#C88).

# In[42]:


# 1. Create the imputer
imputer = SimpleImputer(missing_values=np.nan, strategy= 'mean')


# In[43]:


# 2. Fit the imputer to the numeric colums
imputer.fit(prices_df[numeric_cols])


# In[44]:


# 3. Transform and replace the numeric columns
inputs_df[numeric_cols] = imputer.transform(inputs_df[numeric_cols])


# After imputation, none of the numeric columns should contain any missing values.

# In[45]:


missing_counts = inputs_df[numeric_cols].isna().sum().sort_values(ascending=False)
missing_counts[missing_counts > 0] # should be an empty list


# ## Impute Categorical Data
# Some of the categorical columns in our dataset contain missing values (nan).

# In[46]:


missing_counts = inputs_df[categorical_cols].isna().sum().sort_values(ascending=False)
missing_counts[missing_counts > 0]


# In[47]:


# 1. Create the imputer
cat_imputer = SimpleImputer(missing_values=np.nan, strategy= 'most_frequent')


# In[48]:


# 2. Fit the imputer to the numeric columns
cat_imputer.fit(prices_df[categorical_cols])


# In[49]:


# 3. Transform and replace the numeric columns
inputs_df[categorical_cols] = cat_imputer.transform(inputs_df[categorical_cols])


# In[50]:


# missing_counts = inputs_df[categorical_cols].isna().sum().sort_values(ascending=False)
missing_counts[missing_counts > 0] 


# ### Scale Numerical Values
# 
# The numeric columns in our dataset have varying ranges. 

# In[51]:


inputs_df[numeric_cols].describe().loc[['min', 'max']]


# A good practice is to [scale numeric features](https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range) to a small range of values e.g. $(0,1)$. Scaling numeric features ensures that no particular feature has a disproportionate impact on the model's loss. Optimization algorithms also work better in practice with smaller numbers.
# 

# > **QUESTION 6**: Scale numeric values to the $(0, 1)$ range using `MinMaxScaler` from `sklearn.preprocessing`.
# >
# > *Hint*: See [this notebook](https://jovian.ai/aakashns/python-sklearn-logistic-regression/v/66#C104).

# In[52]:


from sklearn.preprocessing import MinMaxScaler


# In[53]:


# Create the scaler
scaler = MinMaxScaler()


# In[54]:


# Fit the scaler to the numeric columns
scaler.fit(prices_df[numeric_cols])


# In[55]:


# Transform and replace the numeric columns
inputs_df[numeric_cols] = scaler.transform(inputs_df[numeric_cols])


# After scaling, the ranges of all numeric columns should be $(0, 1)$.

# In[56]:


inputs_df[numeric_cols].describe().loc[['min', 'max']]


# ### Encode Categorical Columns
# 
# Our dataset contains several categorical columns, each with a different number of categories.

# In[57]:


inputs_df[categorical_cols].nunique().sort_values(ascending=False)


# In[58]:


from sklearn.preprocessing import OneHotEncoder


# In[59]:


# 1. Create the encoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[60]:


# 2. Fit the encoder to the categorical colums
encoder.fit(inputs_df[categorical_cols])


# In[61]:


# 3. Generate column names for each category
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
len(encoded_cols)


# In[62]:


# 4. Transform and add new one-hot category columns
inputs_df[encoded_cols] = encoder.transform(inputs_df[categorical_cols])


# The new one-hot category columns should now be added to `inputs_df`.

# In[63]:


inputs_df


# Let's save our work before continuing.

# In[64]:


jovian.commit()


# ### Training and Validation Set
# 
# Finally, let's split the dataset into a training and validation set. We'll use a randomly select 25% subset of the data for validation. Also, we'll use just the numeric and encoded columns, since the inputs to our model must be numbers. 

# In[65]:


from sklearn.model_selection import train_test_split


# In[66]:


train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs_df[numeric_cols + encoded_cols], 
                                                                        targets, 
                                                                        test_size=0.25, 
                                                                        random_state=42)


# In[67]:


train_inputs


# In[68]:


train_targets


# In[69]:


val_inputs


# In[70]:


val_targets


# ## Step 3 - Train a Linear Regression Model
# 
# We're train the model. Linear regression is a commonly used technique for solving [regression problems](https://jovian.ai/aakashns/python-sklearn-logistic-regression/v/66#C6). In a linear regression model, the target is modeled as a linear combination (or weighted sum) of input features. The predictions from the model are evaluated using a loss function like the Root Mean Squared Error (RMSE).
# 
# 
# Here's a visual summary of how a linear regression model is structured:
# 
# <img src="https://i.imgur.com/iTM2s5k.png" width="480">
# 
# However, linear regression doesn't generalize very well when we have a large number of input columns with co-linearity i.e. when the values one column are highly correlated with values in other column(s). This is because it tries to fit the training data perfectly. 
# 
# Instead, we'll use Ridge Regression, a variant of linear regression that uses a technique called L2 regularization to introduce another loss term that forces the model to generalize better. Learn more about ridge regression here: https://www.youtube.com/watch?v=Q81RR3yKn30

# In[71]:


from sklearn.linear_model import Ridge


# In[72]:


# Create the model
model = Ridge()


# In[73]:


# Fit the model using inputs and targets
model.fit(train_inputs, train_targets)


# Let's save our work before continuing.

# ## Step 4 - Make Predictions and Evaluate Your Model
# 
# The model is now trained, and we can use it to generate predictions for the training and validation inputs. We can evaluate the model's performance using the RMSE (root mean squared error) loss function.

# In[74]:


from sklearn.metrics import mean_squared_error


# In[75]:


train_preds = model.predict(train_inputs)


# In[76]:


train_preds


# In[77]:


train_rmse = mean_squared_error(train_targets, train_preds, squared=False)


# In[78]:


print('The RMSE loss for the training set is $ {}.'.format(train_rmse))


# In[79]:


val_preds = model.predict(val_inputs)


# In[80]:


val_preds


# In[81]:


val_rmse = mean_squared_error(val_targets, val_preds, squared=False)


# ### Feature Importance
# 
# Let's look at the weights assigned to different columns, to figure out which columns in the dataset are the most important.

# In[82]:


weights = model.coef_


# In[83]:


weights_df = pd.DataFrame({
    'columns': train_inputs.columns,
    'weight': weights
}).sort_values('weight', ascending=False)


# In[84]:


weights_df


# ### Making Predictions
# 
# The model can be used to make predictions on new inputs using the following helper function:

# In[85]:


import numpy as np


# In[86]:


def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[categorical_cols] = cat_imputer.transform(input_df[categorical_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols].values)
    X_input = input_df[numeric_cols + encoded_cols]
    return model.predict(X_input)[0]


# In[87]:


sample_input = {  'MSSubClass': 20, 'MSZoning': 'RL', 'LotFrontage': 77.0, 'LotArea': 9320,
 'Street': 'Pave', 'Alley': None, 'LotShape': 'IR1', 'LandContour': 'Lvl', 'Utilities': 'AllPub',
 'LotConfig': 'Inside', 'LandSlope': 'Gtl', 'Neighborhood': 'NAmes', 'Condition1': 'Norm', 'Condition2': 'Norm',
 'BldgType': '1Fam', 'HouseStyle': '1Story', 'OverallQual': 4, 'OverallCond': 5, 'YearBuilt': 1959,
 'YearRemodAdd': 1959, 'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 'Exterior1st': 'Plywood',
 'Exterior2nd': 'Plywood', 'MasVnrType': 'None','MasVnrArea': 0.0,'ExterQual': 'TA','ExterCond': 'TA',
 'Foundation': 'CBlock','BsmtQual': 'TA','BsmtCond': 'TA','BsmtExposure': 'No','BsmtFinType1': 'ALQ',
 'BsmtFinSF1': 569,'BsmtFinType2': 'Unf','BsmtFinSF2': 0,'BsmtUnfSF': 381,
 'TotalBsmtSF': 950,'Heating': 'GasA','HeatingQC': 'Fa','CentralAir': 'Y','Electrical': 'SBrkr', '1stFlrSF': 1225,
 '2ndFlrSF': 0, 'LowQualFinSF': 0, 'GrLivArea': 1225, 'BsmtFullBath': 1, 'BsmtHalfBath': 0, 'FullBath': 1,
 'HalfBath': 1, 'BedroomAbvGr': 3, 'KitchenAbvGr': 1,'KitchenQual': 'TA','TotRmsAbvGrd': 6,'Functional': 'Typ',
 'Fireplaces': 0,'FireplaceQu': np.nan,'GarageType': np.nan,'GarageYrBlt': np.nan,'GarageFinish': np.nan,'GarageCars': 0,
 'GarageArea': 0,'GarageQual': np.nan,'GarageCond': np.nan,'PavedDrive': 'Y', 'WoodDeckSF': 352, 'OpenPorchSF': 0,
 'EnclosedPorch': 0,'3SsnPorch': 0, 'ScreenPorch': 0, 'PoolArea': 0, 'PoolQC': np.nan, 'Fence': np.nan, 'MiscFeature': 'Shed',
 'MiscVal': 400, 'MoSold': 1, 'YrSold': 2010, 'SaleType': 'WD', 'SaleCondition': 'Normal'}


# In[88]:


predicted_price = predict_input(sample_input)


# In[89]:


print('The predicted sale price of the house is ${}'.format(predicted_price))


# Change the values in `sample_input` above and observe the effects on the predicted price. 

# ### Saving the model
# 
# Let's save the model (along with other useful objects) to disk, so that we use it for making predictions without retraining.

# In[90]:


import joblib


# In[91]:


house_price_predictor = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}


# In[92]:


joblib.dump(house_price_predictor, 'house_price_predictor.joblib')


# Congratulations on training and evaluating your first machine learning model using `scikit-learn`! Let's save our work before continuing. We'll include the saved model as an output.

# ## Make Submission
# 
# To make a submission, just execute the following cell:

# In[93]:


jovian.submit('zerotogbms-a1')


# In[ ]:




