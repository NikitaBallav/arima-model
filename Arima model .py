#!/usr/bin/env python
# coding: utf-8

# # ARIMA model

# A class of statistical model for analyzing and forecasting time series data.
# ### ARIMA stands for AutoRegressive Integrated Moving Average
# 
# Data shows evidence of non-stationarity.
# 
# A random variable that is a time series(random process) is stationary if its statistical properties are all constant over the time.

# - A stationary series has no trend, its variations around its mean have a constant amplitude, and it wiggles in a consistent fashion.
# - The latter condition menas that its autocorrelation remain constant over time.
# - A random variable of this form can be viewed as a combination of signal and noise.
# - An ARIMA model can be viewed as a "filter" that tries to separate the signal from the noise, and the signal is then extrapolated into the future to obtain forecasts.

# ### What is ARIMA forecasting equation for a stationary time series?

# A linear equation in which the predictors consist of lags of the dependent variable and lags of the forecast errors

# ### AR= AutoRegressive
# - uses the dependent relationship between an observation and some number of lagged observations.
# - p = lag order
# 

# ### I= Integrated
# - The use of differencing of raw obseravtions (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary
# - d = Degree of differencing

# ### MA= Moving Averages
# - uses the dependency between an observation and residual errors from a moving average model applied to lagged observations.
# - q = Order of moving average

# ### Assumptions of ARIMA model
# - Stationarity
# - Uncorrelated random error
# - No outliers
# - random shocks (a random error component) if any random error is present they are supposed to be randomly distributed with mean of 0 and constant variance

# ### ARIMA model is also known as Box-Jenkins Method
# - a stochastic model building process

# ### Steps:
# - 1. Visualize the Time series data
# - 2. indentification
# - 3. Estimation
# - 4. diagnostic Checking

# ### Step 2: Identification
# - Assess whether the time series is stationary, and if not, how many differences are required to make it stationary
# - Identify the parameters of an ARMA model for the data

# Unit Root tests- to determine whether or not it is stationary.
# 
# Avoid over differencing 
# 
# configuring AR and MA:
# 
# two diagnostic plots can be used to choose p and q parameter:
# - ACF : AutoCorrelation Function- The plot summarizes the correlation of an observation with lag values. The x-axis shows the lag and the y-axis shows the correlation coefficient between -1 and 1 for negative and positive correlation.
# - PACF : Partial Autocorrelation Function- the plot summarizes the correlation for an observation with lag values that is not accounted for by prior lagged observations.
# 

# The model is AR if the ACF trails off after a lag and has a hard cut-off in the PACF after a lag. This lag is taken as p
# 
# The model is MA if the PACF trails off after a lag and has a hard cut-off in the ACF after a lag. This lag is taken as q
# 
# The model is a mix of AR and MA if both the ACF and PACF trail off.

# ### Step 2: Estimation
# Estimation involves using numerical methods to minimize a loss or error term.

# ### Step 3: Diagnostic Checking
# Look for evidence that the model is not a good fit for the data
# 
# The two areas to investigate diagnostic are overfitting and residual errors
# - Overfitting: The model is more complex than it needs to be and captures random noise in the training data. It negatively impacts the ability of the model to genralized, resulting in poor forecast performance on out of sample data. Careful attention must be paid to both in-sample and out-sample performance.
# - Residuals errors: forecasting the residuals, visualize it with various plots, an ideal model would leave no temporal structure in the time series of forecast residuals.
# 

# # Using ARIMA to predict Bitcoin prices

# In[42]:


# importing the required libraries

import numpy as np, matplotlib.pyplot as plt, pandas as pd
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[70]:


# reading the bitcoin file

df= pd.read_csv("BTC-USD.csv")
df.head()


# In[71]:


df=df.dropna()
df=df.reset_index(drop=True)
df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.shape


# In[72]:


btc=df.set_index("Date")
btc.head()


# In[7]:


x=btc.index
x_sampled=x[::10]
x_sampled


# In[8]:


y=btc["Adj Close"]
y_sampled=y[::10]
y_sampled


# In[9]:


plt.figure(figsize=(20,6))
plt.plot(x_sampled,y_sampled)
plt.xticks(rotation=90)
plt.show


# In[10]:


# train test split
# using 80% of the data as training set and 20% as testing data set

to_row=int(len(btc)*0.8)
print("Selecting the 90% of the rows: ",to_row)
training_data=list(btc[0:to_row]["Adj Close"])
testing_data=list(btc[to_row:]["Adj Close"])


# In[11]:


# the graphical format of train and test sets

plt.figure(figsize=(20,6))
plt.grid(True)
plt.xlabel("Dates")
plt.ylabel("Closing Prices")
plt.plot(btc[0:to_row]["Adj Close"],"green",label="Train data")
plt.plot(btc[to_row:]["Adj Close"], "blue", label="Test data")
plt.xticks(rotation=90)
plt.legend()


# In[12]:


# getting better visualizing by sampling every 10th data point

train_d=btc[0:to_row]["Adj Close"]
test_d=btc[to_row:]["Adj Close"]

x_train=train_d[::10]
x_test=test_d[::10]

plt.figure(figsize=(20,6))
plt.grid(True)
plt.xlabel("Dates")
plt.ylabel("Closing Prices")
plt.plot(x_train,"green",label="Train data")
plt.plot(x_test, "blue", label="Test data")
plt.xticks(rotation=90)
plt.legend()


# By visualizing we get to know that data is not stationary

# In[13]:


# Testing for stationarity

from statsmodels.tsa.stattools import adfuller
test_result = adfuller(btc["Adj Close"])
test_result


# In[40]:


# H0: It is not stationary
# H1: It is stationary

def adfuller_test(x):
    result=adfuller(x)
    labels=["ADF Test statistic","p-value","Lags used","No.of observations used"]
    for value,label in zip(result,labels):
        print(label+':'+str(value))
    if result[1]<=0.05:
        print(" strong evidence to reject null hypothesis i.e. accepting H1 : It is stationary.")
    else:
        print(" weak evidence to reject null hypothesis i.e. accepting H0 : It is not stationary.")


# In[15]:


adfuller_test(btc["Adj Close"])


# Now, we need to make the data stationary by differencing

# In[19]:


# Differencing
# since, the data is non-seasonal we will the use shift(1), if we have got a seasonal data then shift(12)

btc["Adj Close"].shift(1)


# In[20]:


btc["Closing First difference"]=btc["Adj Close"]-btc["Adj Close"].shift(1)


# In[21]:


btc


# In[ ]:


# dropping the nan value and checking the dickey fuller test again


# In[25]:


adfuller_test(btc["Closing First difference"].dropna())


# In[27]:


btc["Closing First difference"].plot()


# We converted our data into stationary.

# In[33]:


# ACF and PACF
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig=plt.figure(figsize=(12,18))
ax1=fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(btc["Closing First difference"].iloc[2:],lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(btc["Closing First difference"].iloc[2:],lags=40,ax=ax2)



# From ACF we can conclude the value of q i.e. for MA it is 1.  
# And from PACF the value of p i.e for AR it is 1. 
# And the difference shift is taken as 1 i.e. d=1
# 

# In[63]:


# p=1, q=1, d=1

from statsmodels.tsa.arima.model import ARIMA

model_predictions=[]
n_test_obser=len(testing_data)
for i in range(n_test_obser):
    model=ARIMA(training_data,order=(1,1,1))
    model_fit=model.fit()
    output=model_fit.forecast()
    yhat=list(output)[0]
    model_predictions.append(yhat)
    actual_test_value=testing_data[i]
    #print(output)
    #break
    training_data.append(actual_test_value)


# In[65]:


print(model_fit.summary())


# In[67]:


# Visualizing the model and predicting the values of the test obs.

plt.figure(figsize=(15,9))
plt.grid(True)
date_range=btc[to_row:].index

plt.plot(date_range,model_predictions,color="blue",marker="o",linestyle="dashed",label="BTC predicted price")
plt.plot(date_range,testing_data,color="red",label="BTC Actual price")

plt.title("Bitcoin Price Predictions")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# In[69]:


mape=np.mean(np.abs(np.array(model_predictions-np.array(testing_data))/np.abs(testing_data)))
print("Mean Absolute Percentage error: "+str(mape))
             


# This implies the model is about 97.1% accurate in predicting the test set observations.

# In[ ]:




