#!/usr/bin/env python
# coding: utf-8

# ### Before starting to play with dataset, you should always have clear concept of what is your OBJECTIVE
# 
#  Once you have clear view of what you want to achieve from this data, it will be far easy to analyze data 
# 
# 
# 

# -----------------

# ## For this demo, My objective it to find the relationship between "age" and "absences" 

# -----------------

# In[4]:



#importing numpy pandas mathplotand other methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[5]:


#first read the data file
df= pd.read_csv('student-por.csv')


# In[6]:


# quick look to dataset
df.head(33)


# In[7]:


#Print summary statistics
df.describe()


# In[8]:


#the complete information about the dataset
df.info()


# In[9]:




# Inspect missing values in the dataset
print(df.isnull().values.sum())

# Replace the ' 's with NaN
df = df.replace(" ",np.NaN)


# Impute the missing values with mean imputation
df = df.fillna(df.mean())

# Count the number of NaNs in the dataset to verify
print(df.isnull().values.sum())


# We see that this data do not contain any missing value
#  ### we are good to go for EDA Cycle
# 

# In[16]:


# seaborn histogram 
sns.distplot(df['age'], hist=True, kde=False, 
             bins=12, color = 'blue',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Age count')
plt.xlabel('Age')

plt.ylabel('Count')



#  The above figure showing histogram of  age count

# In[17]:


# seaborn Scatterplot
sns.scatterplot(x=df['absences'], y=df['age'])


# The above figure showing scatterplot between age and absences
# 

# In[18]:


# line plot 
sns.lineplot(x='absences',y='age', data=df )


#  The above figure is line pot between age and absences. 
#  Blue shade showing confidence interval

# In[19]:


#box plot
sns.boxplot(y = 'age',data= df, x= 'sex')
plt.xlabel('SEX')


#  The above figure showing box plot between age and sex
# 
#  Seen that female are highly concentrated in the age of 16-18 and male are also in same concentrate
# 
#  Median is at 17 age
# 

# In[20]:


#this is heat map pearson correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corrmat, vmax=.8, square=True);


#  The above figure is pearson correlation matrix which shows "How each column are corelated to each other
# 
#  Here, Light color i.e see right scale 0.8 is highly corelated and darker color below -0.2 is not corelated
# 
#  This helps in feature selection also 
# 
# 
# ### If you feel hard to look the color and see which one is high correlated you can do next style 
# 

# In[21]:


plt.figure(figsize=(20,20))
plt.title('Pearson Correlation of Features', size = 15)
colormap = sns.diverging_palette(10, 220, as_cmap = True)
sns.heatmap(df.corr(),
            cmap = colormap,
            square = True,
            annot = True,
            linewidths=0.1,vmax=1.0, linecolor='white',
            annot_kws={'fontsize':12 })
plt.show()



#  In above correlation matrix, we printed the number also so it will be easy for us to see which are highly corelated
#  Value close to 1.00 is highly corelated 
# 
# ### As per our objective,we see that "age" and "absences" have value 0.15 which is below 0.25 and we can say that they are not so correlated
# 
# 

# -----------------

# As objective is set, we complete our EDA Cycle

# ------

# Do you want one magical line of code that can build entire EDA cycle for you: Checkout my [medium article](https://medium.com/analytics-vidhya/quick-exploratory-data-analysis-pandas-profiling-421cd3ec5a50)

# Happy Reading!
