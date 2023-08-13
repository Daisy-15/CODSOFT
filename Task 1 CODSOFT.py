#!/usr/bin/env python
# coding: utf-8

# # Problem

# The problem is to predict whether a passenger survived or not based on various features such as age, sex, class, etc.

# # Importing Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Collection and Preprocessing

# In[3]:


data = pd.read_csv('titanic survival data.csv')
data.head()


# In[4]:


data.info()


# In[5]:


data_description = data.describe()
data_description 


# In[6]:


data.isna()


# In[7]:


sns.heatmap(data.isna())
#visualising missing values, cabin has max null values.


# In[8]:


data.isna().sum()


# In[9]:


# filling mean values of fare and age
fare_mean = data_description.loc['mean', 'Fare']
data['Fare'].fillna(fare_mean, inplace = True)


# In[10]:


age = data['Age']
age_mean = data_description.loc['mean', 'Age']
age.fillna(age_mean, inplace = True)


# In[11]:


age.isna().sum()


# In[12]:


# dropping irrelevant features
data.drop(['Cabin','PassengerId','Name','Ticket'], axis=1, inplace=True)


# In[13]:


data.head()


# # Data visualization 

# In[14]:


data.hist(bins=20, figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[15]:


# visualizing count of survivors and non-survivors using a bar plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=data, palette='Set2')
plt.title('Count of Survivors (0: Not Survived, 1: Survived)')
plt.show()


# In[16]:


# analyzing count of male and female survivors
sns.countplot(x = 'Sex', hue='Survived', data=data)


# In[17]:


plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=data, palette='flare')
plt.title('Survival Distribution by Passenger Class')
plt.show()


# In[18]:


data['Embarked'].value_counts()
# 2-S, 0-C, 1-Q


# In[19]:


data['Sex'].value_counts() 
# male - 1, female - 0


# In[20]:


# convert categorical data to numerical data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])
data['Sex'] = label_encoder.fit_transform(data['Sex'])


# In[21]:


data


# # Separating data

# In[22]:


x = data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = data['Survived']


# # Data Splitting

# In[23]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# # Data Modelling

# In[24]:


#applying logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)


# In[25]:


predict = lr.predict(x_test)


# In[26]:


#testing


# In[27]:


from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])


# In[35]:


from sklearn.metrics import classification_report,accuracy_score,accuracy_score
print(classification_report(y_test,predict))
accuracy = accuracy_score(y_test,predict)
print('Accuracy is: ',accuracy)

