#!/usr/bin/env python
# coding: utf-8

# # **<center><font size=6, color=orange>Linear Regression (Term project)</center>**
# <center> <b>Home Sweet Home - Predicts the rental price of accommodation </center> 

# ---
# # **Table of Contents**
# ---
# 
# **1.** [**Introduction**](#Section1)<br>
# **2.** [**Problem Statement**](#Section2)<br>
# **3.** [**Installing & Importing Libraries**](#Section3)<br>
#   - **3.1** [**Installing Libraries**](#Section31)
#   - **3.2** [**Upgrading Libraries**](#Section32)
#   - **3.3** [**Importing Libraries**](#Section33)
# 
# **4.** [**Data Acquisition & Description**](#Section4)<br>
#   - **4.1** [**Data Description**](#Section41)<br>
# 
# **5.** [**Data Pre-processing**](#Section5)<br>
#   - **5.1** [**Pre-Profiling Report**](#Section51)<br>
#   - **5.2** [**Post-Profiling Report**](#Section52)<br>
# 
# **6.** [**Exploratory Data Analysis**](#Section6)<br>
# **7.** [**Data Post-Processing**](#Section7)<br>
#   - **7.1** [**Feature Extraction**](#Section71)<br>
#   - **7.2** [**Feature Transformation**](#Section72)<br>
#   - **7.3** [**Feature Scaling**](#Section73)<br>
#   - **7.4** [**Data Preparation**](#Section74)<br>
# 
# **8.** [**Model Development & Evaluation**](#Section8)<br>
# **9.** [**Conclusion**](#Section9)<br>
# 

# ---
# <a name = Section1></a>
# # **1. Introduction**
# ---
# 
# <center><img width=50% src='https://www.insaid.co/wp-content/uploads/2021/10/logo.jpg'><b><font size=9>Home Sweet Home <b></font></center>
# 
# <b>Company Introduction</b><br>
# 
# Your client for this project is an online marketplace for lodging, primarily homestays for vacation rentals, and tourism activities.
# 
# - Home Sweet Home (HSH) allows hosts to rent their homestays to other people as guests.
# - The company acts as a mediatory service for the same and has more than 80,000 hosts across 19 cities.
# - Their goal is to provide the best hospitality service to their customers in a more unique and personalized manner.
# 
# <b>Current Scenario</b><br>
# - The company is planning to introduce a new system that will help to easily monitor and predict the rental prices of homes across various cities.

# ---
# <a name = Section2></a>
# # **2. Problem Statement**
# ---
# 
# <b>The current process suffers from the following problems:</b><br>
# 
# - The company monitors and validates the prices set by the hosts.
# - The process of validation is based on various factors such as <b>city, neighborhood, neighborhood group, location on map, availability, and reviews</b>.
# - It is <b>time-consuming, resource-consuming,</b> and sometimes <b>inaccurate</b> to estimate the proper price based on so many factors.
# 
# They have hired you as a data science consultant. They want to supplement their analysis and prediction with a more feasible and accurate approach.
# 
# <b>Your Role</b><br>
# -  are given a historical dataset that contains the price of rental homes and many factors that determine that price.
# - Your task is to build a regression model using the dataset.
# - Because there was no machine learning model for this problem in the company, you don’t have a quantifiable win condition. You need to build the best possible model.
# 
# <b>Project Deliverables</b><br>
# - Deliverable: <b>Predicts the rental price of accommodation.</b>
# - Machine Learning Task: <b>Regression</b>
# - Target Variable: <b>price</b>
# - Win Condition: <b>N/A (best possible model)</b>
# 
# <b>Evaluation Metric</b><br>
# - The model evaluation will be based on the <b>RMSE</b> score.

# ---
# <a name = Section3></a>
# # **3. Installing & Importing Libraries**
# ---

# <a name = Section31></a>
# ### **3.1 Installing Libraries**

# In[1]:


# !pip install -q datascience                                                       # Package that is required by pandas profiling
# !pip install -q pandas-profiling  


# <a name = Section32></a>
# ### **3.2 Upgrading Libraries**

# In[2]:


# !pip install -q --upgrade pandas-profiling


# <a name = Section33></a>
# ### **3.3 Importing Libraries**

# In[3]:


#-------------------------------------------------------------------------------------------------------------------------------
import pandas as pd                                                 # Importing for panel data analysis
from pandas_profiling import ProfileReport                          # Import Pandas Profiling (To generate Univariate Analysis) 
#-------------------------------------------------------------------------------------------------------------------------------
import numpy as np                                                  # Importing package numpys (For Numerical Python)
#-------------------------------------------------------------------------------------------------------------------------------
import plotly.express as px
import matplotlib.pyplot as plt                                     # Importing pyplot interface using matplotlib
import seaborn as sns                                               # Importin seaborm library for interactive visualization
get_ipython().run_line_magic('matplotlib', 'inline')
#-------------------------------------------------------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression                   # Importing Linear Regression model
from sklearn.metrics import mean_squared_error                      # To calculate the MSE of a regression model
from sklearn.metrics import mean_absolute_error                     # To calculate the MAE of a regression model
from sklearn.metrics import r2_score                                # To calculate the R-squared score of a regression model
from sklearn.model_selection import train_test_split                # To split the data in training and testing part
from sklearn.preprocessing import StandardScaler                    # Importing Standard Scaler library from preprocessing
from sklearn.preprocessing import LabelEncoder                      # Importing Label Encoder library from preprocessing
#-------------------------------------------------------------------------------------------------------------------------------
import folium                                                       # Importing folium package
from folium import Map, Marker                                      # Importing folium to plot locations on map
#-------------------------------------------------------------------------------------------------------------------------------
import warnings                                                     # Importing warning to disable runtime warnings
warnings.filterwarnings('ignore')                                   # Warnings will appear only once


# ---
# <a name = Section4></a>
# # **4. Data Acquisition & Description**
# ---

# In[4]:


HomeSweetHome = pd.read_csv('C:/Users/Mahesh/Downloads/Python/Term Projects/ML_Intermediate/rentel price/train_data.csv')
print('Data Shape:', HomeSweetHome.shape)
HomeSweetHome.head()


# <a name = Section41></a>
# ### **4.1 Data Description**
# 
# - In this section we will get **description** and **statistics** about the data.

# Dataset Feature Description
# 
# The Dataset contains the following columns:
# 
# |Column Name|Description|
# |--|--|
# |host_id|	unique host Id|
# |host_name|	name of the host|
# |neighbourhood_group|	group in which the neighbourhood lies|
# |neighbourhood|	name of the neighbourhood|
# |latitude|	latitude of listing|
# |longitude|	longitude of listing|
# |room_type|	type of room|
# |minimum_nights|	minimum no. of nights required to book.|
# |number_of_reviews|	total number of reviews on the listing|
# |last_review|	the date on which listing received its last review|
# |reviews_per_month|	average reviews per month on listing|
# |calculated_host_listings_count|	total number of listings by host|
# |availability_365|	number of days in the year the listing is available for rent|
# |city|	region of the listing|
# |price|	price of listing per night|

# In[5]:


HomeSweetHome.columns


# In[6]:


HomeSweetHome.describe(include='all')


# **Observations**:
# 
# - **minimum_nights** for some home stay can range from as **low** as a **1** to as **high** as **1250**.
# 
# - **price** for some home stay can range from as **low** as a **0** to as **high** as **24999**.
# 
# - **25% of price** have around **75**.
# 
# - **50% of price** have around **120**.
# 
# - **75% of price** have around **200**.

# <a name = Section42></a>
# ### **4.2 Data Information**
# 
#  - In this section, we will get **information about the data** and see some observations.
# 

# In[7]:


HomeSweetHome.info()


# **Observations**:
# 
# - Out of 16 features, we have **1 int64 datatype** features(id), **7 object type** features (name, host_name, 'neighbourhood_group','neighbourhood','room_type','last_review','city'), and the **rest are of float64** datatype features.
# 
# - We may have to **convert some variables** like **('minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'price')** into appropriate forms so we can use them for training purposes.

# <a name = Section5></a>
# 
# ---
# # **5. Data Pre-Processing**
# ---

# <a name = Section51></a>
# ### **5.1 Pre Profiling Report**

# In[8]:


profile = ProfileReport(rental_df, title="Rental Profiling Report")
profile.to_file("Rental_report.html")
print('Accomplished!')


# <b><font size=4, color=orange> Observations from Profile Report</b>
# ---
#    
# |Observations|Values|
# |:--|:--|
# |Number of columns|17|
# |Number of rows|137023|
# |Missing cells|116342|
# |Duplicate rows|0|
# |Continuous type columns|10|
# |Categorical type columns|7 |
# 
# ---
# 
# #### B. Missing Data from below variables:<p>
#     
# |Observations|Values|
# |:--|:--|
# |name|16|
# |host_name|21|
# |neighbourhood_group|56723|
# |last_review|29791|
# |reviews_per_month|29791|
# 
# ---
# 
# #### C. Below are unique values:<p>
# 
# |Observations|Values|
# |:--|:--|
# |neighbourhood_group |17|
# |room_type |4|
# |city|19|
# 
# ---

# <b><font size=4, color=orange> Performing Operations

# In[9]:


HomeSweetHome.isna().sum()


# In[10]:


# lets check data for Nul values in 'reviews_per_month'

HomeSweetHome[HomeSweetHome['reviews_per_month'].isna()]


# In[11]:


HomeSweetHome[HomeSweetHome['neighbourhood_group'].isna()]['city'].value_counts()


# In[12]:


HomeSweetHome["neighbourhood_group"].value_counts(dropna=False)


# <b> <font size=4, color=orange>Observation: </b>
# 1. we can see the the null values for 'last review' & 'reviews per month' are also null because of  'No of Reviews' is Zero
# 2. we can fill null values in 'reviews per month' with 0
# 3. 'Neighbourhood_group' missing values wil be filled with 'City'
# 4. 'last_review' contains Date and it may not be valauble in projecting price while Training the Model, so it can be droped.

# In[13]:


HomeSweetHome["reviews_per_month"].fillna(value=0, inplace=True)
HomeSweetHome["neighbourhood_group"]=np.where(HomeSweetHome["neighbourhood_group"].isnull(),
                                              HomeSweetHome['city'],HomeSweetHome["neighbourhood_group"])


# In[14]:


print(HomeSweetHome.shape)


# In[15]:


HomeSweetHome[HomeSweetHome.duplicated()]


# <a name = Section6></a>
# 
# ---
# # **6. Exploratory Data Analysis**
# ---

# ## 6.1 Univariate Analysis

# In[16]:


HomeSweetHome.city.value_counts()


# In[17]:


# to check distribution of 'neighbourhood_group'

px.histogram(HomeSweetHome, x= 'neighbourhood_group', marginal='box',
                nbins=47, title='Distribution of neighbourhood_group')


# In[18]:


# to check distribution of 'room_type'

px.histogram(HomeSweetHome, x= 'room_type', marginal='box',
                nbins=47, title='Distribution of room_type')


# In[19]:


# to check distribution of 'minimum_nights'

plt.figure(figsize=(15,3))
sns.distplot(HomeSweetHome['minimum_nights'])


# In[20]:


# to check distribution of 'number_of_reviews'

plt.figure(figsize=(15,3))
sns.distplot(HomeSweetHome['number_of_reviews'])


# In[21]:


# to check distribution of 'calculated_host_listings_count'

plt.figure(figsize=(15,3))
sns.distplot(HomeSweetHome['calculated_host_listings_count'])


# In[22]:


# to check distribution of 'availability_365'

plt.figure(figsize=(15,3))
sns.distplot(HomeSweetHome['availability_365'])


# In[28]:


px.histogram(HomeSweetHome, x= 'availability_365', marginal='box',
                title='Distribution of availability_365')


# In[31]:


px.histogram(HomeSweetHome, x= 'price', marginal='box',
                nbins=47, title='Distribution of price')


# In[32]:


px.histogram(HomeSweetHome[HomeSweetHome['price']<1000], x= 'price', marginal='box',
                nbins=47, title='Distribution of price')


# <b> <font size=4, color=orange>Observation: </b>
# 1. In 'Manhatten', 'City of Los Angeles' and 'Brooklyn' - Neighbourhood_Groups are more than 10000 stay available.
# 2. There are more than 80000 Entire Home/ Apt properties are listed.
# 3. 'Minimm Nights', 'Number of Reviews' and 'calculated_host_listings_count' having large nummber of outliers, as all these are skewed towards right.
# 4. 'Availibilty_365' seems noramlly distributed.
# 5. Price is distributed mostly in range between 0 to 400.
# 6. There are 0 values also in - 'Number of Reviews', 'calculated_host_listings_count' , 'Availibilty_365' and 'Minimum Nights'

# ## 6.2 Bivariate analysis

# In[33]:


px.scatter(HomeSweetHome,x='number_of_reviews',y='reviews_per_month')


# In[34]:


px.scatter(HomeSweetHome,x='calculated_host_listings_count',y='price')


# In[35]:


px.scatter(HomeSweetHome,y='calculated_host_listings_count',x='availability_365')


# In[36]:


px.scatter(HomeSweetHome,x='reviews_per_month',y='price')


# In[37]:



plt.figure(figsize=(6,4))
sns.barplot(data=HomeSweetHome,x='room_type',y='price')


# ## 6.3 Multivariate analysis

# In[38]:



plt.figure(figsize=(8,8))
sns.heatmap(HomeSweetHome.corr(),annot=True,cmap='icefire')


# In[39]:


HomeSweetHome.skew()


# <b> <font size=4, color=orange>Observation: </b>
# 1. There is slight relation between 'reviews_per_month' and 'number_of_reviews'
# 2. 'Hotel Room' are highly priced than others.
# 3. Greater the price lesser the reviews.
# 4. 'Minimum_nights', 'number_of_reviews', 'number_of_reviews', 'calculated_host_listings_count' and 'price' are Higly positive skewed.
# 5. 'id' is Correraled with 'latitude' and 'longitude'
# 6. 'latitude' is Correraled with 'longitude'
# 7. 'reviews_per_month' is Correraled with 'number_of_reviews'
# 8. 'calculated_host_listings_count' is Correraled with 'availability_365'

# <a name = Section7></a>
# 
# ---
# # **7. Data Post-Processing**
# ---

# <a name = Section71></a>
# ### **7.1 Feature Extraction**

# <b> Categegorical & Continuous Varibale Split </b>

# In[40]:


HomeSweetHome.info()


# In[41]:


HomeSweetHome_copy=HomeSweetHome.copy()


# In[42]:


HomeSweetHome_copy[['minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count',
                   'availability_365','price']]=HomeSweetHome_copy[['minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count',
                   'availability_365','price']].astype('int')


# In[43]:


HomeSweetHome_copy.info()


# In[44]:


#Filtering the data without 0 Values in Price

HomeSweetHome_copy=HomeSweetHome_copy[HomeSweetHome_copy['price']>0]


# In[45]:


HomeSweetHome_copy.shape


# In[46]:


HomeSweetHome_copy=HomeSweetHome_copy[HomeSweetHome_copy['price']<400]
HomeSweetHome_copy.shape


# In[47]:


df_cont = HomeSweetHome_copy.select_dtypes(exclude='object')

df_cat = HomeSweetHome_copy.select_dtypes(include='object')


# In[48]:


df_cat.head()


# In[49]:


df_cont.head()


# <b> Dropping the un-necessary features</b>

# In[50]:


# 'name', 'host_name', 'neighbourhood', 'last_review' can be dropped from Categorical df
df_cat=df_cat.drop(['name', 'host_name', 'neighbourhood', 'last_review'], axis=1)

# 'id','host_id','reviews_per_month' can be dropped from Continuous df
df_cont=df_cont.drop(['id','host_id','reviews_per_month'],axis=1)


# In[51]:


df_cont.head()


# In[52]:


df_cat.head()


# <a name = Section72></a>
# ### **7.2 Feature Transformation**

# In[53]:


df_cat = df_cat.apply(LabelEncoder().fit_transform)
df_cat.head()


# <b> Combining both Catgorical & Continous df </b>

# In[54]:


df_comb=pd.concat([df_cat, df_cont], axis = 1)
df_comb.head()


# In[55]:


df_comb.skew()


# <a name = Section73></a>
# ### **7.3 Feature Independant & Dependant Separation**

# In[56]:


def seperate_Xy(data=None):
    X = data.drop(labels=['price'], axis=1)
    y = data['price']
    return X, y


# In[57]:


X, y = seperate_Xy(data=df_comb)
X.head()


# In[58]:


y.head()


# <a name = Section73></a>
# ### **7.3 Feature Scaling**
# 
# - In this section, we will perform **standard scaling** over the selected features.

# In[59]:


scaler = StandardScaler() 
scaler.fit(X) 

X = pd.DataFrame(scaler.transform(X), index = X.index, columns = X.columns + '_S')


# In[60]:


X.head()


# In[61]:


y = np.log(y)
y.head()


# In[62]:


X.skew()


# <a name = Section74></a>
# ### **7.4 Data Preparation for Training & Testing**

# In[63]:


def Xy_splitter(X=None, y=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    print('Training Data Shape:', X_train.shape, y_train.shape)
    print('Testing Data Shape:', X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


# In[64]:


X_train, X_test, y_train, y_test = Xy_splitter(X=X, y=y)


# <a name = Section8></a>
# 
# ---
# # **8. Model Development & Evaluation**
# ---

# In[65]:


def model_generator_lr():
    return LinearRegression()


# In[66]:


clf = model_generator_lr()


# In[67]:


def train_n_eval(clf=None):
    
    # Extracting model name
    model_name = type(clf).__name__
    
    # Fit the model on train data
    clf.fit(X_train, y_train)
    
    # Make predictions using test data
    y_pred = clf.predict(X_test)
    
    # Make predictions using test data
    y_pred_train = clf.predict(X_train)
    
    # Calculate test accuracy of the model
    clf_mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate train accuracy of the model
    clf_mae_train = mean_absolute_error(y_train, y_pred_train)
    
    # Calculate test accuracy of the model
    clf_mse = mean_squared_error(y_test, y_pred)
    
    # Calculate train accuracy of the model
    clf_mse_train = mean_squared_error(y_train, y_pred_train)
    
    # Calculate test accuracy of the model
    clf_r2 = r2_score(y_test, y_pred)
    
    # Calculate train accuracy of the model
    clf_r2_train = r2_score(y_train, y_pred_train)
    
    # Display the accuracy of the model
    print('Performance Metrics for', model_name, ':\n')
    print('[Mean Absolute Error Train]:', clf_mae_train)
    print('[Mean Absolute Error Test]:', clf_mae, ':\n')
    print('*******************************\n')
    print('[Mean Sqaured Error Train]:', clf_mse_train)
    print('[Mean Sqaured Error Test]:', clf_mse, ':\n')
    print('*******************************\n')
    print('[Root Mean Sqaured Error Train]:', np.sqrt(clf_mse_train))
    print('[Root Mean Sqaured Error Test]:', np.sqrt(clf_mse), ':\n')
    print('*******************************\n')
    print('[R2-Score Train]:', clf_r2_train)
    print('[R2-Score Test]:', clf_r2)


# In[68]:


train_n_eval(clf=clf)


# <b> <font size=4, color=orange>Observation: </b>
# 
#     
#     
#     

# # Implementing Model on provided data

# In[69]:


df_test=pd.read_csv('C:/Users/Mahesh/Downloads/Python/Term Projects/ML_Intermediate/rentel price/test_data.csv')
df_test.head()


# In[70]:


df_test["reviews_per_month"].fillna(value=0, inplace=True)
df_test["neighbourhood_group"]=np.where(df_test["neighbourhood_group"].isnull(),df_test['city'],df_test["neighbourhood_group"])

df_test[['minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count',
                   'availability_365']]=df_test[['minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count',
                   'availability_365']].astype('int')
                   
df_test_cont = df_test.select_dtypes(exclude='object')

df_test_cat = df_test.select_dtypes(include='object')

df_test_cat=df_test_cat.drop(['name', 'host_name', 'neighbourhood', 'last_review'], axis=1)

df_test_cont=df_test_cont.drop(['id','host_id','reviews_per_month'],axis=1)


df_test_cat = df_test_cat.apply(LabelEncoder().fit_transform)

df_test_comb=pd.concat([df_test_cat, df_test_cont], axis = 1)

scaler.fit(df_test_comb) 

df_test_comb = pd.DataFrame(scaler.transform(df_test_comb), index = df_test_comb.index, columns = df_test_comb.columns + '_S')


# In[71]:


df_test.head()


# In[72]:


# Make predictions using test data
y_test_pred = clf.predict(df_test_comb)
print(np.round(np.exp(y_test_pred),decimals=0))


# In[73]:


index=df_test['id']
result=pd.DataFrame(np.round(np.exp(y_test_pred),decimals=0),index=index)


# In[74]:


result


# In[75]:


result.to_csv('C:/Users/Mahesh/Downloads/Python/Term Projects/ML_Intermediate/rentel price/result.csv', header=False)
print("Successful")

