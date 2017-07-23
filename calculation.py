
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from datetime import datetime
import re
from dateutil.parser import parse


# In[2]:

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


# In[3]:

def weatherclean(filename):
    file=pd.read_csv(filename, skiprows=25)
    name=re.findall('[a-z]*[0-9]+', filename)
    file.drop(['Year', 'Month', 'Day', 'Data Quality','Mean Temp Flag', 'Heat Deg Days (Â°C)', 'Heat Deg Days Flag', 'Min Temp Flag',
       'Cool Deg Days (Â°C)', 'Cool Deg Days Flag','Total Rain Flag','Total Snow Flag','Total Precip Flag','Snow on Grnd Flag',
               'Dir of Max Gust Flag','Spd of Max Gust Flag', 'Total Snow (cm)', 'Total Rain (mm)', 'Max Temp Flag', 'Snow on Grnd (cm)'],
              1,inplace=True)
    file.columns=['Date', 'Max Tem', 'Min Temp', 'Mean Temp', 'Total Precip', 'Dir of Wind', 'Spd of Wind']
    for i in range(len(file)):
        file.set_value(i, 'Date', parse(file['Date'][i]).date()) # Turn string into datetime and only show date
    globals()[name[0]]=file   # This automatically make a variable named after the csv file and assign the DF to that variable


# In[4]:

weatherclean('past_weather/w1991.csv')
weatherclean('past_weather/w1992.csv')
weatherclean('past_weather/w1993.csv')
weatherclean('past_weather/w1994.csv')
weatherclean('past_weather/w1995.csv')
weatherclean('past_weather/w1996.csv')
weatherclean('past_weather/w1997.csv')
weatherclean('past_weather/w1998.csv')
weatherclean('past_weather/w1999.csv')
weatherclean('past_weather/w2000.csv')
weatherclean('past_weather/w2001.csv')
weatherclean('past_weather/w2002.csv')
weatherclean('past_weather/w2003.csv')
weatherclean('past_weather/w2004.csv')
weatherclean('past_weather/w2005.csv')
weatherclean('past_weather/w2006.csv')
weatherclean('past_weather/w2007.csv')
weatherclean('past_weather/w2008.csv')
weatherclean('past_weather/w2009.csv')
weatherclean('past_weather/w2010.csv')
weatherclean('past_weather/w2011.csv')
weatherclean('past_weather/w2012.csv')
weatherclean('past_weather/w2013.csv')
weatherclean('past_weather/w2014.csv')
weatherclean('past_weather/w2015.csv')
weatherclean('past_weather/w2016.csv')
weatherclean('past_weather/w2017.csv')


# In[5]:

def lotclean(filename):
    file=pd.read_csv(filename)
    for i in range(len(file)):
        file.set_value(i, 'Date', parse(file['Date'][i]).date()) # Turn string into datetime and only show date
    name=re.findall('[a-z]*[0-9]+', filename)
    globals()['l'+name[0]]=file


# In[6]:

lotclean('past_numbers/1991.csv')
lotclean('past_numbers/1992.csv')
lotclean('past_numbers/1993.csv')
lotclean('past_numbers/1994.csv')
lotclean('past_numbers/1995.csv')
lotclean('past_numbers/1996.csv')
lotclean('past_numbers/1997.csv')
lotclean('past_numbers/1998.csv')
lotclean('past_numbers/1999.csv')
lotclean('past_numbers/2000.csv')
lotclean('past_numbers/2001.csv')
lotclean('past_numbers/2002.csv')
lotclean('past_numbers/2003.csv')
lotclean('past_numbers/2004.csv')
lotclean('past_numbers/2005.csv')
lotclean('past_numbers/2006.csv')
lotclean('past_numbers/2007.csv')
lotclean('past_numbers/2008.csv')
lotclean('past_numbers/2009.csv')
lotclean('past_numbers/2010.csv')
lotclean('past_numbers/2011.csv')
lotclean('past_numbers/2012.csv')
lotclean('past_numbers/2013.csv')
lotclean('past_numbers/2014.csv')
lotclean('past_numbers/2015.csv')
lotclean('past_numbers/2016.csv')
lotclean('past_numbers/2017.csv')


# In[7]:

weatherlist=[w1991,w1992,w1993,w1994,w1995,w1996,w1997,w1998,w1999,w2000, w2001, w2002, w2003, w2004, w2005, w2006, w2007, w2008, w2009, w2010, w2011, w2012, w2013, w2014,
            w2015, w2016,w2017]
lotlist=[l1991,l1992,l1993,l1994,l1995,l1996,l1997,l1998,l1999,l2000, l2001, l2002, l2003, l2004, l2005, l2006, l2007, l2008, l2009, l2010, l2011, l2012, l2013, l2014, l2015,
        l2016, l2017]


# In[8]:

weather_data=pd.concat(weatherlist)
lot_data=pd.concat(lotlist)


# In[9]:

weather_data.reset_index(inplace=True)


# In[10]:

weather_data.drop('index',1,inplace=True)


# In[11]:

summary=weather_data.merge(lot_data, on='Date', how='inner')


# ##### Cleaning up some nan for temperature and precipitation columns

# In[15]:

for i in range(len(summary)):
    if np.isnan(summary['Mean Temp'][i]):
        summary.set_value(i, 'Mean Temp', (summary['Mean Temp'][i-1] + summary['Mean Temp'][i-1])/2)


# In[17]:

for i in range(len(summary)):
    if np.isnan(summary['Total Precip'][i]):
        summary.set_value(i, 'Total Precip', (summary['Total Precip'][i-1] + summary['Total Precip'][i-1])/2)


# In[ ]:




# In[ ]:

# summary_clean=summary.dropna()


# In[19]:

meantemp_orig=summary['Mean Temp'].values
totalprecip_orig=summary['Total Precip'].values
Num1_orig=summary['Num1'].values


# In[40]:

Num2_orig=summary['Num2'].values
Num3_orig=summary['Num3'].values
Num4_orig=summary['Num4'].values
Num5_orig=summary['Num5'].values
Num6_orig=summary['Num6'].values


# In[18]:

summary['Total Precip'].isnull().sum()    # Checking how many NaN in the column


# ##### So many missing values for Dir of Wind and Spd of Wind

# In[ ]:

# meantemp=summary_clean['Mean Temp'].values
# totalprecip=summary_clean['Total Precip'].values
# winddir=summary_clean['Dir of Wind'].values
# windspeed=summary_clean['Spd of Wind'].values
# Num1=summary_clean['Num1'].values


# In[20]:

# windspeed=windspeed.astype(np.float)   # Need to turn string into float for wind speed


# In[ ]:

# X_4D=np.column_stack((meantemp, totalprecip, winddir, windspeed))  # Putting all arrays together


# In[21]:

X_2D=np.column_stack((meantemp_orig, totalprecip_orig))


# In[51]:

X_train, X_test, y_train, y_test =train_test_split(X_2D, Num3_orig)


# #### Random Forest on Num1

# In[52]:

clf=RandomForestClassifier(max_features=2, random_state=1)


# In[53]:

clf.fit(X_train,y_train )


# In[54]:

clf.score(X_test, y_test)


# #### Gradient boosted decision tree

# In[55]:

from sklearn.ensemble import GradientBoostingClassifier


# In[56]:

clf=GradientBoostingClassifier().fit(X_train, y_train)


# In[57]:

clf.score(X_test, y_test)


# #### What about neural network?

# In[58]:

nnclf=MLPClassifier(hidden_layer_sizes=[100, 100], solver='lbfgs', random_state=0).fit(X_train, y_train)


# In[59]:

nnclf.score(X_test, y_test)


# #### Dummy classifier?

# In[60]:

from sklearn.dummy import DummyClassifier
dummy_majority=DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
dummy_majority.score(X_test, y_test)


# In[61]:

dummy_majority=DummyClassifier(strategy = 'stratified').fit(X_train, y_train)
dummy_majority.score(X_test, y_test)


# In[62]:

dummy_majority=DummyClassifier(strategy = 'prior').fit(X_train, y_train)
dummy_majority.score(X_test, y_test)


# In[63]:

dummy_majority=DummyClassifier(strategy = 'uniform').fit(X_train, y_train)
dummy_majority.score(X_test, y_test)


# In[ ]:



