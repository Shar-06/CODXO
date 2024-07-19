#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:


#Initialize the datasets
fake_data = pd.read_csv("Fake.csv")
real_data = pd.read_csv("True.csv")

#Create labels for the datasets
fake_data['label'] = "FAKE"
real_data['label'] = "REAL"

#combine the two datasets
data = pd.concat([fake_data,real_data]).reset_index(drop=True)

#Shuffling to randomize the data to avoid biased results
data = data.sample(frac = 1)

#Preprocess data to clean it of unnecessary characters such as punctuations,urls,etc that may confuse the model during training
def preprocess(text):
    text = text.lower() #converts all characters to lowercase for uniformity
    text = re.sub('\[.*?\]','',text) #remove non-alphabetics
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://S+/www\.\S+','',text) #remove urls
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text) #remove punctuations
    text = re.sub('<.*?>+','',text) #remove HTML tags
    text = re.sub('\w*\d\w*','',text) #remove words with digits
    text = re.sub('\n','',text) #remove new lines
    
    return text

#Apply the preprocessing
data['text'] = data['text'].apply(preprocess)

x,y = data['text'], data['label']

#split the dataset for training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)

#Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words = "english", max_df = 0.7)

#vectorize the split sets
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)


# In[4]:


#The LinearSVC classifier
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(x_train_vectorized, y_train)
clf.score(x_test_vectorized, y_test)


# In[5]:


#The LogisticRegression classifier
from sklearn.linear_model import LogisticRegression 
LR = LogisticRegression()
LR.fit(x_train_vectorized, y_train)
LR.score(x_test_vectorized, y_test)


# In[6]:


#The DecisionTree classifier
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(x_train_vectorized, y_train)
DT.score(x_test_vectorized, y_test)


# In[7]:


#The GradientBoosting classifier
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(random_state = 0)
GB.fit(x_train_vectorized, y_train)
GB.score(x_test_vectorized, y_test)


# In[8]:


#The RandomForest classifier
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state = 0)
RF.fit(x_train_vectorized, y_train)
RF.score(x_test_vectorized, y_test)


# In[11]:


#Define a function that will produce the predictions given a text input
def Detect(news):
    news_txt = {'text':[news]} #create a dictionary to prepare input for format conversion to a dataframe
    news_data = pd.DataFrame(news_txt) 
    news_data['text'] = news_data['text'].apply(preprocess)
    new_x_test = news_data['text']
    new_x_test_vectorized = vectorizer.transform(new_x_test)
    
    #classifier predictions
    predict_clf = clf.predict(new_x_test_vectorized)
    predict_lr = LR.predict(new_x_test_vectorized)
    predict_dt = DT.predict(new_x_test_vectorized)
    predict_gb = GB.predict(new_x_test_vectorized)
    predict_rf = RF.predict(new_x_test_vectorized)  
    
    result = "\n\nCLF Prediction: {} \nLR Prediction: {} \nDT Prediction: {} \nGB Prediction: {} \nRF Prediction: {}".format(
           predict_clf[0],
           predict_lr[0],
           predict_dt[0],
           predict_gb[0],
           predict_rf[0]
    )
    
    return result

News = str(input())
print(Detect(News))


# In[12]:


News = str(input())
print(Detect(News))


# In[ ]:




