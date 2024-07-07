#!/usr/bin/env python
# coding: utf-8

# # Lm Python

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# In[3]:


df = pd.read_csv("regrex1.csv")


# In[4]:


df.head


# In[5]:


x_values = df['x'].to_numpy()  
y_values = df['y'].to_numpy()


# In[6]:


lm = stats.linregress(x_values,y_values)


# In[7]:


plt.scatter(x_values,y_values)
plt.savefig("pylmscript.png")

# In[13]:


plt.scatter(x_values,y_values)
plt.plot(x_values,lm.slope*x_values+lm.intercept)
plt.savefig("pypredictscript.png")

# In[61]:




















