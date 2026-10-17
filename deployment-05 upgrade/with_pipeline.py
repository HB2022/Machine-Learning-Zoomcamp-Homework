#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# In[2]:


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


# In[12]:


data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(data_url)

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[13]:


y_train = df.churn


# In[14]:


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


# In[15]:


print(y_train.value_counts())


# In[16]:


pipeline = make_pipeline(
    DictVectorizer(sparse=False),
    LogisticRegression(solver='liblinear')
)


# In[17]:


train_dict = df[categorical + numerical].to_dict(orient='records')
pipeline.fit(train_dict, y_train)


# In[19]:


with open('model.bin', 'wb') as f_out:
    pickle.dump(pipeline, f_out)


# In[20]:


get_ipython().system('ls -lh')


# In[21]:


with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


# In[22]:


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

churn = pipeline.predict_proba(customer)[0, 1]
print('probability of churnning = ', churn)
if churn >= 0.5:
    print('send email')
else: print('dont do anything')


# In[ ]:




