import numpy as np
import pandas_datareader as web
import matplotlib.pyplot as plt
import pandas as pd 
import datetime as datetime
import scipy.optimize as optimization
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



start_date= '01/01/2010'
end_date = '06/01/2020'
stocks = ['LT.NS']


def download_data(stocks,start_date,end_date,lags = 5):
	data = web.DataReader(stocks,'yahoo', start_date, end_date)
	datalag = pd.DataFrame(index = data.index)
	datalag['Today']= data['Adj	Close']
	datalag['Volume'] = data['Volume']


	for i in range (0,lags):
		datalag['lags%s' %str(i+1)] = data['Adj Close'].shift(i+1)
# 
	dfret = pd.DataFrame(index = data.index)
	dfret['Volume']= datalag['Volume']
	dfret['Today'] = datalag['Today'].pct_change()*100

	for i in range(0,lags):
		dfret['lags%s' %str(i+1)] = datalag['lags%s'%str(i+1)].pct_change()*100

	dfret['Direction'] = np.sign(dfret['Today'])

	dfret.drop(dfret.index[:5],inplace = True)
	return dfret




# Create a lagged series of the S&P500 US stock market index
data1 = download_data(stocks, start_date, end_date, lags=5)
print(data.head())

# # Use the prior two days of returns as predictor 
# # values, with direction as the response
# X = data[["Lag1","Lag2","Lag3","Lag4"]]
# y = data["Direction"]

# # The test data is split into two parts: Before and after 1st Jan 2005.
# start_test = datetime(2017,1,1)

# # Create training and test sets
# X_train = X[X.index < start_test]
# X_test = X[X.index >= start_test]
# y_train = y[y.index < start_test]
# y_test = y[y.index >= start_test]

# #we use Logistic Regression as the machine learning model
# model = LogisticRegression()                              

# #train the model on the training set
# model.fit(X_train, y_train)

# #make an array of predictions on the test set
# pred = model.predict(X_test)

# #output the hit-rate and the confusion matrix for the model
# print("Accuracy of logistic regression model: %0.3f" % model.score(X_test, y_test))
# print("Confusion matrix: \n%s" % confusion_matrix(pred, y_test))







