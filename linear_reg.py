
# coding: utf-8

# In[15]:


import os
os.chdir('/home/infinityvoyager/sagar/datascience/datasets/30_sept')


# In[16]:


import pandas as pd
import numpy as np


# In[17]:


df = pd.read_csv('train.csv')
print(df.shape)
print(df.dtypes)


# In[18]:


df['SalePrice'].head()


# In[19]:


list(df)


# In[20]:


correlation_values = df.select_dtypes(include=[np.number]).corr()
correlation_values


# In[22]:


correlation_values[['SalePrice']]


# In[28]:


#df1 = df[((df['SalePrice']>0.6)&(df['SalePrice']<1)) & ((df['SalePrice']<(-0.6))&(df['SalePrice']>(-1)))]


# In[38]:


#(df['SalePrice']>=0.6)&(df['SalePrice']<1)


# In[39]:


selected_features = correlation_values[["SalePrice"]][(correlation_values["SalePrice"]>=0.6)|(correlation_values["SalePrice"]<=-0.6)]


# In[44]:


selected_features


# In[52]:


selected_features.drop('SalePrice',axis=0)


# In[73]:


X = df[['OverallQual','TotalBsmtSF','GrLivArea','GarageArea','1stFlrSF']]


# In[74]:


y = df['SalePrice']


# In[75]:


from sklearn.model_selection import train_test_split as tts


# In[76]:


X_train, X_test, y_train, y_test = tts(X,y,test_size = 0.3, random_state = 42)

y_pred = reg.predict(X_test)
# In[77]:


from sklearn.linear_model import LinearRegression


# In[78]:


reg = LinearRegression()


# In[79]:


reg.fit(X_train, y_train)


# In[80]:


y_pred = reg.predict(X_test)


# In[81]:


reg.score(X_test, y_test)


# In[82]:


from sklearn.metrics import mean_squared_error


# In[83]:


rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rmse

