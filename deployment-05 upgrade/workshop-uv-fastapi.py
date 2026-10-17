#!/usr/bin/env python
# coding: utf-8

# This is a starter notebook for an updated module 5 of ML Zoomcamp
# 
# The code is based on the modules 3 and 4. We use the same dataset: [telco customer churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

# In[5]:


import pandas as pd
import numpy as np
import sklearn


# In[6]:


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


# In[7]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[8]:


data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(data_url)

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[9]:


y_train = df.churn


# In[10]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[11]:


dv = DictVectorizer()

train_dict = df[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict).toarray()
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)


# In[12]:


customer = {'gender': 'male',
 'seniorcitizen': 0,
 'partner': 'no',
 'dependents': 'yes',
 'phoneservice': 'no',
 'multiplelines': 'no_phone_service',
 'internetservice': 'dsl',
 'onlinesecurity': 'no',
 'onlinebackup': 'yes',
 'deviceprotection': 'no',
 'techsupport': 'no',
 'streamingtv': 'no',
 'streamingmovies': 'no',
 'contract': 'month-to-month',
 'paperlessbilling': 'yes',
 'paymentmethod': 'electronic_check',
 'tenure': 6,
 'monthlycharges': 29.85,
 'totalcharges': 129.85}


# In[13]:


X = dv.transform(customer)


# In[14]:


churn = model.predict_proba(X)[0, 1]


# In[15]:


churn


# In[1]:


import pickle


# In[17]:


with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# In[2]:


with open('model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


# model

# In[7]:


customer = {'gender': 'male',
 'seniorcitizen': 0,
 'partner': 'no',
 'dependents': 'yes',
 'phoneservice': 'no',
 'multiplelines': 'no_phone_service',
 'internetservice': 'dsl',
 'onlinesecurity': 'no',
 'onlinebackup': 'yes',
 'deviceprotection': 'no',
 'techsupport': 'no',
 'streamingtv': 'no',
 'streamingmovies': 'no',
 'contract': 'month-to-month',
 'paperlessbilling': 'yes',
 'paymentmethod': 'electronic_check',
 'tenure': 6,
 'monthlycharges': 29.85,
 'totalcharges': 129.85}
X = dv.transform(customer)
churn = model.predict_proba(X)[0, 1]
if churn >= 0.5:
    print('send email')
else: print('dont do anything')


# In[8]:


churn


# In[ ]:




