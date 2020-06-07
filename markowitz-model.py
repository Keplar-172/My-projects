import numpy as np
import pandas_datareader as web
import matplotlib.pyplot as plt
import pandas as pd 
import datetime as datetime
import scipy.optimize as optimization

# 'ADANIPORTS.NS','M&M.NS','RELIANCE.NS','SETFNIF50.NS' Optimal weights : [0.    0.    0.357 0.466 0.    0.    0.041 0.135 0.   ]
df = pd.read_csv('holdings.csv')
df['total inv'] = df['Avg. cost']*df['Qty.']
df['weights'] = df['total inv']/df['total inv'].sum()
stocks = df['Instrument']
stocks = stocks +'.NS'
# stocks = ['HDFCBANK.NS','GAIL.NS','HINDUNILVR.NS','TATACHEM.NS','TVSMOTOR.NS']
start_date= '01/01/2010'
end_date = '06/01/2020'


# Downloading data
def download_data(stocks):
	data = web.DataReader(stocks,'yahoo', start_date, end_date)['Adj Close']
	data.columns = stocks
	return data

def show_data(data):
	data.plot(figsize = (10,5))
	plt.show()

# Calculation daily returns
def daily_return(data):
	returns = np.log(data/data.shift(1))
	return returns
# plot daily returns
def plot_daily_returns(retunrs):
	retunrs.plot(figsize=(10,5))
	plt.show()

def show_stat(returns):
	print(returns.mean()*252)
	print(returns.cov()*252)

# Weights
def initialize_weights():
	weights = np.random.random(len(stocks))
	weights/= np.sum(weights)
	return weights

# calculate portfolio return
def calculate_portfolio_return(returns, weights):
	portfolio_return = np.sum(returns.mean()*weights)*252
	print('Expected portfolio return : ', portfolio_return)

# calculate portfolio variance

def calculate_portfolio_variance(returns,weights):
	portfolio_variance = np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
	print('Expected variance',portfolio_variance)
	return portfolio_variance
# Generate portfolio
# def generate_portfolios(weights,returns):

def generate_portfolios(weights,returns):
	preturns=[]
	pvariance=[]

	# Montecarlo simulation
	for i in range(10000):
		weights= np.random.random(len(stocks))
		weights/= np.sum(weights)
		preturns.append(np.sum(returns.mean()*weights)*252)
		pvariance.append(np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights))))
	preturns = np.array(preturns)
	pvariance = np.array(pvariance)
	return preturns, pvariance

# plot portfolios
def plot_portfolios(returns,variances):
	plt.figure(figsize=(10,6))
	plt.scatter(variances,returns,c = returns/variances,marker ='o')
	plt.grid(True)
	plt.xlabel('Expected volatality')
	plt.ylabel('Expected returns')
	plt.colorbar(label= 'Sharpe Ratio')
	plt.show()
# getting statistics of all portfolios
def statistics(weights, returns):
	portfolio_return =np.sum(returns.mean()*weights)*252
	portfolio_volatality = np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252, weights)))
	return np.array([portfolio_return,portfolio_volatality,portfolio_return/portfolio_volatality])

# getting min sharpe ratio
def min_funs_sharpe(weights,returns):
	return -statistics(weights, returns)[2]

def optimise_portfolio(weights,returns):
	constraints =({'type':'eq','fun':lambda x :np.sum(x)-1})
	bounds = tuple((0,1) for x in range(len(stocks)))
	optimum = optimization.minimize(fun = min_funs_sharpe,x0= weights,args = returns,method = 'SLSQP',bounds = bounds, constraints = constraints)
	return optimum

def print_optimum_portfoilo(optimum,returns):
	print('Optimal weights :', optimum['x'].round(3))
	print('Expected return, volatality, sharpe ratio : ',statistics(optimum['x'].round(3),returns))

def show_optimal_portfolio(optimum,returns,preturns,pvariance):
	plt.figure(figsize=(10,6))
	plt.scatter(pvariance,preturns,c = preturns/pvariance,marker ='o')
	plt.grid(True)
	plt.xlabel('Expected volatality')
	plt.ylabel('Expected returns')
	plt.colorbar(label= 'Sharpe Ratio')
	plt.plot(statistics(optimum['x'],returns)[1],statistics(optimum['x'],returns)[0],'g*',markersize = 20.0)
	plt.show()














data= download_data(stocks)
show_data(data)
returns = daily_return(data)
plot_daily_returns(returns)
show_stat(returns)
weights = initialize_weights()
calculate_portfolio_return(returns, weights)
calculate_portfolio_variance(returns,weights)
preturns, pvariance = generate_portfolios(weights,returns)
plot_portfolios(preturns,pvariance)
optimum =optimise_portfolio(weights,returns)
print_optimum_portfoilo(optimum,returns)
show_optimal_portfolio(optimum,returns,preturns,pvariance)

