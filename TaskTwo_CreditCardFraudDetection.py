#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Read the data
credit_data = pd.read_csv("creditcard.csv")
credit_data


# In[2]:


#Check the distribution of legit and fraudulent transactions
#If the dataset is unbalanced, it may be biased towards a certain class
credit_data['Class'].value_counts()


# In[3]:


Legit_trans = credit_data[credit_data.Class == 0]
Fraud_trans = credit_data[credit_data.Class == 1]

#Statistical measures(mean,average,stddev,etc) of the Legit transactions
Legit_trans.Amount.describe()


# In[4]:


Fraud_trans.Amount.describe()


# In[5]:


#Compare statistical values for legit and fraudulent transactions
credit_data.groupby('Class').mean()


# In[6]:


#Since there's more of legit transactions, make the transactions of legit data be equal to that of fraudulent transactions
Legit_sample = Legit_trans.sample(n = 492)

#Combine the sample data with the fraudulent set, axis = 0 means (rows) added one after another 
sample_data = pd.concat([Legit_sample, Fraud_trans], axis = 0)


# In[7]:


#Check the distribution
sample_data['Class'].value_counts()


# In[8]:


#Check the mean of a uniform dataset
sample_data.groupby('Class').mean()


# In[9]:


#Split the sample dataset for training and testing, axis = 1 means (columns) being dropped/removed
x = sample_data.drop(columns = 'Class', axis = 1) #We want the columns with transactions values not the Class column
y = sample_data['Class']

#We stratify the train and testing to ensure the distribution of the data in x_train and x_test is uniform, we stratify to y as the labels in the training and testing data can be different
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, stratify = y, random_state = 2) 


# In[10]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(x_train, y_train)
#No longer need to vectorize as we are dealing with numericals

#Model perfomance evaluation on training data
x_train_predict = LR.predict(x_train)
training_accuracy = accuracy_score(x_train_predict, y_train)
print("Training Accuracy :",training_accuracy)
print(classification_report(y_train, x_train_predict))


# In[11]:


#Model performance evaluation on testing data
x_test_predict = LR.predict(x_test)
testing_accuracy = accuracy_score(x_test_predict, y_test)
print("Testing Accuracy :",testing_accuracy)
print(classification_report(y_test, x_test_predict))


# In[ ]:




