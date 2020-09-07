
# coding: utf-8



# In[1]:

import pandas as pd


# In[2]:

data = pd.read_csv('bank_note_data.csv')




# In[3]:

data.head()


# ## EDA


# In[4]:

import seaborn as sns
get_ipython().magic('matplotlib inline')



# In[5]:

sns.countplot(x='Class',data=data)




# In[6]:

sns.pairplot(data,hue='Class')


# ## Data Preparation 

# ### Standard Scaling
# 
# ** 

# In[7]:

from sklearn.preprocessing import StandardScaler



# In[8]:

scaler = StandardScaler()



# In[9]:

scaler.fit(data.drop('Class',axis=1))



# In[10]:

scaled_features = scaler.fit_transform(data.drop('Class',axis=1))



# In[11]:

df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()


# ## Train Test Split
# 
#

# In[12]:

X = df_feat


# In[13]:

y = data['Class']



# In[14]:

X = X.as_matrix()
y = y.as_matrix()



# In[15]:

from sklearn.cross_validation import train_test_split


# In[16]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# In[17]:

import tensorflow.contrib.learn as learn
import tensorflow as tf



# In[18]:

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
classifier = learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10, 20, 10], n_classes=2)



# In[19]:

classifier.fit(X_train, y_train, steps=200, batch_size=20)


# ## Model Evaluation
# 


note_predictions = classifier.predict(X_test)



# In[21]:

from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
data=np.array(list(note_predictions))


# In[22]:

print(confusion_matrix(y_test,data))


# In[23]:

print(classification_report(y_test,data))


# ## Comparison
# 

# In[24]:

from sklearn.ensemble import RandomForestClassifier


# In[25]:

rfc = RandomForestClassifier(n_estimators=200)


# In[26]:

rfc.fit(X_train,y_train)


# In[27]:

rfc_preds = rfc.predict(X_test)


# In[28]:

print(classification_report(y_test,rfc_preds))


# In[29]:

print(confusion_matrix(y_test,rfc_preds))


