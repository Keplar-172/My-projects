import pandas as pd 
import numpy as np
import pandas_datareader as web



# stocks =['LT.NS','RELIANCE.NS']
# weights = [0.25,0.75]
start_date = '01/01/2010'
end_date = '06/01/2020'

def get_data(stocks):
	data= web.DataReader(stocks, 'yahoo',start_date,end_date)['Adj Close']
	return data

# data = get_data(stocks)
def daily_return(data):
	returns = np.log(data/data.shift(1))
	return returns
# returns =daily_return(data)
# print(returns)
# data = pd.concat([data,returns],axis =1)
# print(data)
# print(data['LT.NS'])



def portfolio_return(weights, returns):
	portfolio_return = np.sum(returns.mean()*weights*252)
	print('expected portfolio return : ', portfolio_return)
	return portfolio_return

def portfolio_variance(weights, returns):
	portfolio_variance = np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
	print('expected portfolio variance : ',portfolio_variance)
	return portfolio_variance

def calculate_sharpe(portfolio_return,portfolio_variance):
	sharpe = portfolio_return/portfolio_variance
	return sharpe


# portfolio_return =portfolio_return(weights,returns)






df = pd.read_csv('holdings.csv')
df['total inv'] = df['Avg. cost']*df['Qty.']
df['weights'] = df['total inv']/df['total inv'].sum()
stocks = df['Instrument']
stocks = stocks +'.NS'
weights = df['weights'].to_numpy()

# weights.reset_index

data = get_data(stocks)
returns = daily_return(data)
print(weights)
portfolio_return = portfolio_return(weights,returns)
portfolio_variance= portfolio_variance(weights,returns)
sharpe = calculate_sharpe(portfolio_return,portfolio_variance)
print('sharpe value : ' ,sharpe)
