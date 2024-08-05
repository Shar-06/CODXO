#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[29]:


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


# In[32]:


#The LogisticRegression classifier
from sklearn.linear_model import LogisticRegression 
LR = LogisticRegression()
LR.fit(x_train_vectorized, y_train)
predict_lr = LR.predict(x_test_vectorized)
LR.score(x_test_vectorized, y_test)


# In[33]:


print(classification_report(y_test, predict_lr))


# In[34]:


#Define a function that will produce the predictions given a text input
def Detect(news):
    news_txt = {'text':[news]} #create a dictionary to prepare input for format conversion to a dataframe
    news_data = pd.DataFrame(news_txt) 
    news_data['text'] = news_data['text'].apply(preprocess)
    new_x_test = news_data['text']
    new_x_test_vectorized = vectorizer.transform(new_x_test)
    
    #classifier prediction
    predict_lr = LR.predict(new_x_test_vectorized)
    
    result = "\n\nLR Prediction: {}".format(
           predict_lr[0]
    )
    
    return result

News = str(input())
print(Detect(News))


# In[35]:


News = str(input())
print(Detect(News))


# In[ ]:




