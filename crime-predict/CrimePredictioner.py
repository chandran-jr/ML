#!/usr/bin/env python
# coding: utf-8

# # CRIME PREDICTION USING ML

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
df=pd.read_csv('crime.csv')
df


# # DROPPING UNWANTED COLUMNS

# In[2]:


dataset = pd.DataFrame(df)
dataset = dataset.drop(['YEAR','MONTH','DAY','HOUR','MINUTE','HUNDRED_BLOCK','NEIGHBOURHOOD','X','Y','Latitude'], axis=1	)
dataset


# # A DEMO VIEW OF 30 ROWS

# In[3]:


df.head(30)


# # DEMO DATA

# In[4]:


demo_view = df.iloc[1]
demo_view


# # REDUCING SIZE OF DATASET

# **INDEPENDENT VARIABLE**

# In[5]:


x=df.head(2000)
x


# **LATITUDE**

# In[6]:


a = x.iloc[:,10:11].values
a


# **LONGITUDE**

# In[7]:


b = x.iloc[:,11].values
b


# # TAKING AVERAGE OF LATITUDE AND LONGITUDE

# In[8]:


X=[]
for i in range(0,2000):
    ab = a[i]+b[i]
    X.append(ab/2)

X


# # DEPENDENT VARIABLE

# **REDUCING SIZE TO 2K**

# In[9]:


y=df.head(2000)
y


# **DEPENDENT VARIABLE TAKEN**

# In[10]:


y = y.iloc[:,0:1].values
y


# # FINDING DIFFERENT TYPES OF CRIME

# In[11]:


YY = [] 
ynum=0
for i in y: 
    if i not in YY: 
        ynum+=1
        YY.append(i)

YY


# **THERE ARE 7 DIFFERENT TYPES OF CRIMES IN THE DATASET**

# In[12]:


ynum


# # REPLACING EACH CRIME WITH A NUMBER FOR EASY COMPUTATION

# In[13]:


Y=y

for i in range(0,2000):
    if Y[i]=='Other Theft':
        Y[i]=1
    elif Y[i]=='Break and Enter Residential/Other':
        Y[i]=2
    elif Y[i]=='Mischief':
        Y[i]=3
    elif Y[i]=='Break and Enter Commercial':
        Y[i]=4
    elif Y[i]=='Offence Against a Person':
        Y[i]=5
    elif Y[i]=='Theft from Vehicle':
        Y[i]=6
    else:
        Y[i]=7

Y
  


# # DATA VISUALIZATION

# In[14]:


X


# **MANY NULL VALUES WERE FOUND**

# # DATA CLEANING

# **THE NULL VALUE ROWS SHOULD BE DROPPED**

# In[15]:


nullvals=0
for i in X:
    if i>=0:
        nullvals+=1

nullvals


# **167 SUCH ROWS WERE FOUND**

# **REMOVING THOSE 167 ROWS IN BOTH X AND Y**

# In[16]:


YY=Y.tolist()
popin=0

for i in X:
     if i>=0:
        X.pop(popin)
        YY.pop(popin)
        popin+=1


# In[17]:


XXvals=0
for i in X:
    XXvals+=1
XXvals


# In[18]:


X


# In[19]:


YYvals=0
for i in YY:
     YYvals+=1

YYvals


# In[20]:


YY = np.array(YY, dtype=np.int)


# In[21]:


YYvals=0
for i in YY:
    YYvals+=1

YYvals


# # FINDING THE NUMBER OF EACH TYPE OF CRIME

# In[22]:


theft=0
breakin=0
mischief=0
commercial=0
offence=0
vehicle=0
collision=0

for i in YY:
    if i == 1:
        theft+=1
    elif i == 2:
        breakin+=1
    elif i == 3:
        mischief+=1
    elif i == 4:
        commercial+=1
    elif i == 5:
        offence+=1
    elif i == 6:
        vehicle+=1
    elif i == 7:
        collision+=1


print("Number of thefts= ", theft)
print("Number of break and Enter Residential/Other = ", breakin)
print("Number of mischief = ", mischief)
print("Number of Break and Enter Commercial= ", commercial)
print("Number of Offence Against a Person = ", offence)
print("Number of Theft from Vehicle = ", vehicle)
print("Number of Vehicle Collision or Pedestrian Struck (with Injury) = ", collision)


# # VIEWING THE CRIME PLOT GRAPH

# **BAR GRAPH**

# In[23]:


labels = ['Theft', 'Residential','Mischief','Commercial','Person','Theft Vehicle', 'Vehicle Collide']
number = [theft,breakin,mischief,commercial,offence,vehicle,collision]

fig, ax = plt.subplots()

width = 0.4

ax.bar(labels, number,width, color='Orange')

ax.set_ylabel('NUMBER', color= 'aqua')
ax.set_title('CRIME NUMBER GRAPH', color='yellow')

plt.show()


# **PIE CHART**

# In[24]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Theft', 'Residential','Mischief','Commercial','Person','Theft Vehicle', 'Vehicle Collide'
sizes = [theft,breakin,mischief,commercial,offence,vehicle,collision]
explode = (0, 0, 0, 0, 0, 0, 0.1) 

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 

plt.show()


# # USING KNN

# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X,YY,test_size=0.30)


# In[27]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[28]:


knn.fit(X_train,y_train)


# In[29]:


pred = knn.predict(X_test)


# In[30]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score


# In[31]:


print(confusion_matrix(y_test,pred))


# In[32]:


print(classification_report(y_test,pred))


# In[33]:


accuracy_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,X,YY,cv=10)
    accuracy_rate.append(score.mean())


# In[34]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,X,YY,cv=10)
    error_rate.append(1-score.mean())


# In[35]:


plt.figure(figsize=(20,12))
#plt.plot(range(1,80),error_rate,color='blue', linestyle='dashed', marker='o',
  #       markerfacecolor='red', markersize=10)
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[36]:


# NOW WITH K=8
knn = KNeighborsClassifier(n_neighbors=32)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=8')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# # CRIME PREDICTING

# **THE DATASET IS CONCENTRATED IN ONE PLACE ONLY DUE TO WHICH THE LATITUDE-LONGITUDE RANGE CAN VARY ONLY FROM -30 to -40**

# In[37]:


print("Give the desired Longitude")
longi = float(input())
print("Give the desired Latitude")
latit = float(input())
longlang = (longi+latit)/2
inp = np.array([longlang]) 
inp = inp.reshape(1, -1)

prediction = knn.predict(inp)

if (prediction == 1):
  print("Crime is Theft")
elif (prediction == 3):
  print("Crime is mischief")
elif (prediction == 2):
  print("Crime is Break and Enter Residential")
elif (prediction == 4):
  print("Crime is Break and Enter commercial")
elif (prediction == 5):
  print("Crime is Offence against a person")
elif (prediction == 6):
  print("Crime is Theft from vehicle")
else:
  print("Crime is Vehicle Collision or pedestrian struck/injured")


# # THE OUTPUT USUALLY WOULD GIVE ONLY "Vehicle Collision or pedestrian struck/injured" BECAUSE THATS THE HIGHEST NUMBER OF CRIME IN THE DATASET. A BETTER DATASET WOULD GIVE BETTER RESULTS
