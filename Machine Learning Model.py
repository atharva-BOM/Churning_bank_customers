#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data = pd.read_csv("D:\VOIS ML\Final Project\Bank_Customer_Churn_dataset (Edited).csv")


# In[3]:


dataset = data.drop(['RowNumber','CustomerId','Surname'],axis=1)


# In[5]:


data.columns


# In[6]:


dataset =  dataset.drop(['Geography', 'Gender'], axis=1)


# In[8]:


Geography = pd.get_dummies(data.Geography).iloc[:,1:]
Gender = pd.get_dummies(data.Gender).iloc[:,1:]


# In[9]:


dataset = pd.concat([dataset,Geography,Gender], axis=1)


# In[10]:


X =  dataset.drop(['Exited'], axis=1)
y = dataset['Exited']


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[13]:


from sklearn.ensemble import RandomForestClassifier


# In[14]:


classifier = RandomForestClassifier(n_estimators=200, random_state=0)  
classifier.fit(X_train, y_train)  
predictions = classifier.predict(X_test)


# In[15]:


from sklearn.metrics import classification_report, accuracy_score


# In[16]:


print(classification_report(y_test,predictions ))  
print(accuracy_score(y_test, predictions ))


# In[17]:


feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')


# In[19]:


feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(4).plot(kind='bar')


# In[ ]:


# Age and EstimatedSalary are the 2 most important deciding factors in predicitng the churn of customers likely to leave the bank


# In[ ]:




