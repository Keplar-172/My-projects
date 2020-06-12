import pandas as pandas
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from pandas_datareader import data, wb

start_date = '01/01/2010'
end_date = '06/01/2020'
risk_free_rate = 0.04

stocks = ['LT.NS','^NSEI']
data = web.DataReader(stocks,'yahoo',start_date,end_date)['Adj Close']
data1 = data.resample('M').last()
# print(data1.head())
# print(data.head())
returns = np.log(data1/data1.shift(1))
returns = returns.dropna()
cov_mat = np.array(returns.cov())
beta = cov_mat[1,0]/cov_mat[1,1]
print(beta,'\n',cov_mat)
beta1,alpha = np.polyfit(returns['^NSEI'],returns['LT.NS'],deg =1)
print(beta1,alpha)

# plot
fig,axis = plt.subplots(1,figsize=(20,10))
axis.scatter(returns['^NSEI'],returns['LT.NS'],label= 'data and points')
axis.plot(returns['^NSEI'],beta1*returns['^NSEI'] + alpha, color = 'red', label= 'CAPM line')
plt.title('CAPM, ALPHA AND BETA')
plt.xlabel('market return $R m$')
plt.ylabel('stock return')
plt.legend()
# plt.text()
plt.grid(True)
plt.show()

expected_return = risk_free_rate +beta1*(returns['^NSEI'].mean()*12 - risk_free_rate)
print('expected_return', expected_return)