#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------DATA CLEANING----------------------------

# read the data
df = pd.read_csv("PMA_dataset.csv")

# Dealing with missing values 
# filling missing values with medians of the columns
df = df.fillna(df.median())
# drop data with missing values
df = df.dropna()

# drop unecessary columns
drop_col=['Applicants total','ID number','Name']
df = df.drop(drop_col, axis = 1)
df


# In[92]:


# ------Data transformation----------------------------
#change yes/no into 1/0
for column in ['Offers Less than one year certificate',
                'Offers One but less than two years certificate',
                'Offers Associate\'s degree',
                'Offers Two but less than 4 years certificate',
                'Offers Bachelor\'s degree',
                'Offers Postbaccalaureate certificate',
                'Offers Master\'s degree',
                'Offers Post-master\'s certificate',
                'Offers Doctor\'s degree - research/scholarship',
                'Offers Doctor\'s degree - professional practice',
                'Offers Doctor\'s degree - other',
                'Offers Other degree']:
    df[column]=df[column].replace(["Yes","No"],[1,0])
    
#----feature scaling-------------------------------
# min/max normalisation
for column in ['Percent of freshmen submitting SAT scores',
               'Percent of freshmen submitting ACT scores',
               'Percent of total enrollment that are American Indian or Alaska Native',
               'Percent of total enrollment that are Asian',
               'Percent of total enrollment that are Black or African American',
               'Percent of total enrollment that are Hispanic/Latino',
               'Percent of total enrollment that are Native Hawaiian or Other Pacific Islander',
               'Percent of total enrollment that are White',
               'Percent of total enrollment that are two or more races',
               'Percent of total enrollment that are Race/ethnicity unknown',
               'Percent of total enrollment that are Nonresident Alien',
               'Percent of total enrollment that are Asian/Native Hawaiian/Pacific Islander',
               'Percent of total enrollment that are women',
               'Percent of undergraduate enrollment that are American Indian or Alaska Native',
               'Percent of undergraduate enrollment that are Asian',
               'Percent of undergraduate enrollment that are Black or African American',
               'Percent of undergraduate enrollment that are Hispanic/Latino',
               'Percent of undergraduate enrollment that are Native Hawaiian or Other Pacific Islander',
               'Percent of undergraduate enrollment that are White',
               'Percent of undergraduate enrollment that are two or more races',
               'Percent of undergraduate enrollment that are Race/ethnicity unknown',
               'Percent of undergraduate enrollment that are Nonresident Alien',
               'Percent of undergraduate enrollment that are Asian/Native Hawaiian/Pacific Islander',
               'Percent of undergraduate enrollment that are women',
               'Percent of graduate enrollment that are American Indian or Alaska Native',
               'Percent of graduate enrollment that are Asian',
               'Percent of graduate enrollment that are Black or African American',
               'Percent of graduate enrollment that are Hispanic/Latino',
               'Percent of graduate enrollment that are Native Hawaiian or Other Pacific Islander',
               'Percent of graduate enrollment that are White',
               'Percent of graduate enrollment that are two or more races',
               'Percent of graduate enrollment that are Race/ethnicity unknown',
               'Percent of graduate enrollment that are Nonresident Alien',
               'Percent of graduate enrollment that are Asian/Native Hawaiian/Pacific Islander',
               'Percent of graduate enrollment that are women',
               'Percent of first-time undergraduates - in-state',
               'Percent of first-time undergraduates - out-of-state',
               'Percent of first-time undergraduates - foreign countries',
               'Percent of first-time undergraduates - residence unknown',
               'Graduation rate - Bachelor degree within 4 years, total',
               'Graduation rate - Bachelor degree within 5 years, total',
               'Graduation rate - Bachelor degree within 6 years, total',
               'Percent of freshmen receiving any financial aid',
               'Percent of freshmen receiving federal, state, local or institutional grant aid',
               'Percent of freshmen  receiving federal grant aid',
               'Percent of freshmen receiving Pell grants',
               'Percent of freshmen receiving other federal grant aid',
               'Percent of freshmen receiving state/local grant aid',
               'Percent of freshmen receiving institutional grant aid',
               'Percent of freshmen receiving student loan aid',
               'Percent of freshmen receiving federal student loans',
               'Percent of freshmen receiving other loan aid',
               'SAT Critical Reading 25th percentile score',
               'SAT Critical Reading 75th percentile score',
               'SAT Math 25th percentile score',
               'SAT Math 75th percentile score',
               'SAT Writing 25th percentile score',
               'SAT Writing 75th percentile score',
               'ACT Composite 25th percentile score',
               'ACT Composite 75th percentile score',
               'Percent admitted - total'
              ]:
    #df[column]=df[column]/100
    df[column]=(df[column]-df[column].min())/(df[column].max()-df[column].min())
df['Percent of freshmen submitting SAT scores']


# In[93]:


# standardization
for column in ['Estimated enrollment, total',
               'Estimated enrollment, full time',
               'Estimated enrollment, part time',
               'Estimated undergraduate enrollment, total',
               'Estimated undergraduate enrollment, full time',
               'Estimated undergraduate enrollment, part time',
               'Estimated freshman undergraduate enrollment, total',
               'Estimated freshman enrollment, full time',
               'Estimated freshman enrollment, part time',
               'Estimated graduate enrollment, total',
               'Estimated graduate enrollment, full time',
               'Estimated graduate enrollment, part time',
               'Associate\'s degrees awarded',
               'Bachelor\'s degrees awarded',
               'Master\'s degrees awarded',
               'Doctor\'s degrese - research/scholarship awarded',
               'Doctor\'s degrees - professional practice awarded',
               'Doctor\'s degrees - other awarded',
               'Certificates of less than 1-year awarded',
               'Certificates of 1 but less than 2-years awarded',
               'Certificates of 2 but less than 4-years awarded',
               'Postbaccalaureate certificates awarded',
               'Post-master\'s certificates awarded',
               'Number of students receiving an Associate\'s degree',
               'Number of students receiving a Bachelor\'s degree',
               'Number of students receiving a Master\'s degree',
               'Number of students receiving a Doctor\'s degree',
               'Number of students receiving a certificate of less than 1-year',
               'Number of students receiving a certificate of 1 but less than 4-years',
               'Number of students receiving a Postbaccalaureate or Post-master\'s certificate',
               'Admissions yield - total',
               'Tuition and fees, 2010-11',
               'Tuition and fees, 2011-12',
               'Tuition and fees, 2012-13',
               'Tuition and fees, 2013-14',
               'Total price for in-state students living on campus 2013-14',
               'Total price for out-of-state students living on campus 2013-14',
               'Total  enrollment',
               'Full-time enrollment',
               'Part-time enrollment',
               'Undergraduate enrollment',
               'Graduate enrollment',
               'Full-time undergraduate enrollment',
               'Part-time undergraduate enrollment',
               'Number of first-time undergraduates - in-state',
               'Number of first-time undergraduates - out-of-state',
               'Number of first-time undergraduates - foreign countries',
               'Number of first-time undergraduates - residence unknown'
              ]:
    df[column]=(df[column]-df[column].mean())/df[column].std()
df['SAT Math 25th percentile score']


# In[94]:


# ------Dimension reduction----------------------------

X=df.values
from sklearn.decomposition import PCA  #import PCA
pca = PCA().fit(X)

plt.plot(np.cumsum(pca.explained_variance_ratio_)) 
#show the relationship between cumulative explained variance and number of components
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
#define xlabel and ylabel


# In[98]:


pca = PCA(n_components=0.99) #explained variance=0.99
newX = pca.fit_transform(X) 
print(pca.explained_variance_ratio_)
#print(newX)
#invX = pca.inverse_transform(X)  


# In[99]:


print(pca.components_)
pca.components_.argmax(axis=1)


# In[106]:


# ------Feature Selection----------------------------

#get unique variable
res = {}.fromkeys(pca.components_.argmax(axis=1)).keys() 
for i in res:
    print(df.columns.values[i])


# In[ ]:




