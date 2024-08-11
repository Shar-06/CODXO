#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

#Read the dataset
Reviews_Data = pd.read_csv("Dataset.csv")

Reviews_Data


# In[7]:


# Cleaning and preprocessing the data
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub('\[.*?\]', '', text)  # Remove content within brackets
    text = re.sub("\\W", " ", text)  # Remove non-alphabetic characters
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub('<.*?>+', '', text)  # Remove HTML tags
    text = re.sub('\w*\d\w*', '', text)  # Remove words containing digits
    text = re.sub('\n', '', text)  # Remove newlines
    
    return ' '.join(text.split())  # Correctly join with single spaces

# Apply cleaning to the text reviews 
Reviews_Data['review'] = Reviews_Data['review'].apply(preprocess)

# Split into training and testing datasets
x = Reviews_Data['review']
y = Reviews_Data['sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the vectorizer
vect = TfidfVectorizer(stop_words="english", max_df=0.7)

# Vectorize the split sets
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)

# Initialize the model classifier
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=500)

# Train the model
LR.fit(x_train_vect, y_train)
training_predict_lr = LR.predict(x_train_vect)
training_accuracy = accuracy_score(training_predict_lr, y_train) 

# Evaluate the model training performance
print("Training Accuracy:", training_accuracy)
print(classification_report(y_train, training_predict_lr))

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(LR, x_train_vect, y_train, normalize="all")
plt.title("Training Confusion Matrix")
plt.show()


# In[10]:


#Test the model
LR.fit(x_test_vect, y_test)
testing_predict_lr = LR.predict(x_test_vect)
testing_accuracy = accuracy_score(testing_predict_lr, y_test)

# Evaluate the model testing performance
print("Testing Accuracy :",training_accuracy)
print(classification_report(y_test, testing_predict_lr))

#Confusion matrix
ConfusionMatrixDisplay.from_estimator(LR, x_test_vect, y_test, normalize = "all")
plt.title("Testing Confusion Matrix")
plt.show()


# In[ ]:




