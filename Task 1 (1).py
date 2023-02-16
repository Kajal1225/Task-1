#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# TASK 1 - Prediction using Supervised ML.
# 
#  To Predict the percentage of marks of the students based on the number of hours they studied
# 
# Author - 

# In[2]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[3]:


#loading the data
data = pd.read_csv('data.txt')
data.head()


# In[7]:


#Checking for null value
data.isnull == True


# No null value,so we can visualize

# In[4]:


sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(data.corr())


# Variables are positively correlated.

# # Training the Model

# 1) Splitting the Data
# 

# In[5]:


# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# 2) Fitting the Data into the model
# 

# In[6]:


regression = LinearRegression()
regression.fit(train_X, train_y)
print("----Model Trained-----")


# # Predicting the Percentage of Marks
# 

# In[7]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# # Comparing the Predicted Marks with the Actual Marks
# 

# In[8]:


compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


# In[9]:


plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[10]:


# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))


# What will be the predicted score of a student if he/she studies for 9.25 hrs/ day?
# 

# In[11]:


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# According to the regression model if a student studies for 9.25 hours a day he/she is likely to score 93.89 marks.
