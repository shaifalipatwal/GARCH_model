#GARCH Model (Generalized Auto Regressive Conditional Heteroskedasticity Model)
pip install pandas-datareader
pip install arch
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf
import yfinance as yf

#%%
# Specifying the sample: I took S&P financial data from Yahoo Financing website
ticker = '^GSPC'
start = '2015-12-31'
end = '2023-06-25'

# Downloading data
prices = yf.download(ticker, start, end)['Close']
prices.shape
returns = prices.pct_change().dropna()
#%%
# Splitting data into train and test sets 
train_size = int(len(returns) * 0.8)  #training on 80% data
train_size
train_returns, test_returns = returns[:train_size], returns[train_size:]
#%%
# Plotting the returns
plt.figure(figsize=(10, 6))
plt.plot(returns.index, returns, color='blue', linestyle='-')
plt.title('Returns of S&P 500 Index', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Returns', fontsize=14)
plt.xticks(rotation=45)
plt.show()

#Here we can see the sudden brust in the data which means there is a sudden jump in the data at some period of time and then it again goes down 
# To solve this brusty problem, we'll use GARCH model

#%%
# Plotting PACF graph for returns 
plot_pacf(returns**2)
#%%
# Fitting GARCH(1,1) model as the p-value for the GARCH(2,2) was not significant
# I tried to fit model at GARCH(2,2) by changing the p and q values in the code. But the beta values were insignificant
# Therefore I am using GARCH (1,1)
model = arch_model(train_returns, p=1, q=1)
model_fit = model.fit()
model_fit.summary()

# The summary shows that the the parameters alpha[1] and beta[1] are significant as the p-values are very low

#%%
# Predicting future values
forecasts = model_fit.forecast(start=len(test_returns), horizon=len(test_returns))
forecasts

# Plotting the predicted values
plt.figure(figsize=(10, 6))
plt.plot(test_returns.index, np.sqrt(forecasts.variance.values[-1, :]), color='red', linestyle='-')
plt.title('Predicted Volatility of S&P 500 Index', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Volatility', fontsize=14)
plt.xticks(rotation=45)
plt.show()
