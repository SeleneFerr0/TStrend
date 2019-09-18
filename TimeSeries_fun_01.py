# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:42:20 2019

@author: SeleneFerro

Functions for Time series forecast and trend break detection

"""

from __future__ import print_function

import xlrd
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os, psutil
import rpy2.robjects as robjects
#from bokeh.io import output_file, show
#from bokeh.plotting import figure
#from numpy import histogram, linspace
from scipy.stats.kde import gaussian_kde
import shelve
import pytz
from datetime import datetime
from pandas import ExcelWriter
import math
import imblearn
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)



from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv


def cnt_var(df, var):
    var_lst = list(df[var].unique())
    print(var)
    for v in var_lst:
        print(str(v) + ' ' + str(len(df[df[var]== v])))


def cnt_per(df, var):
    var_lst = list(df[var].unique())
    print(var)
    for v in var_lst:
        print(str(v) + ' ' + "{:.0%}".format((len(df[df[var]== v])/len(df))))


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)




def seasonal_mean(ts, n, lr=0.7):
    """
    Compute the mean of corresponding seasonal periods
    ts: 1D array-like of the time series
    n: Seasonal window length of the time series
    """
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            ts_seas = ts[i-1::-n]  # previous seasons only
            if np.isnan(np.nanmean(ts_seas)):
                ts_seas = np.concatenate([ts[i-1::-n], ts[i::n]])  # previous and forward
            out[i] = np.nanmean(ts_seas) * lr
    return out


# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))



# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models



def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]






from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv
 
# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]
 
# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = sarima_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error
 
# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)
 
# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores
 
# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models



''' SARIMA FORECAST'''

def sarima_detect(train_set, test_set, shoptype, categories, thresh = 1, tol = 0.4, order=(0,0,0), seasonal_order=(0, 1, 0, 12)):
    global outputs
    outputs = output_dir + shoptype+ '\\'
    dicky_test=pd.DataFrame()
    if not os.path.exists(outputs + '\\broken'):
        os.makedirs(outputs + '\\broken')
    if not os.path.exists(outputs + '\\good'):
        os.makedirs(outputs + '\\good')
    
    for cat in categories:
        x = np.log(train_set[cat]+1)
        y = np.log(test_set[cat]+1)
        ax =plt.gca()
        x.plot(title = cat,colormap='jet')
        y.plot()
        x.rolling(6).mean().plot()
    #    x.interpolate(inplace = True)
        x.index=x.index.to_timestamp()
    #    result_mul = seasonal_decompose(x, model='addtive')
    #    deseasonalized = x / result_mul.seasonal
    #    results_AR = model.fit(disp=-1)
    #    x_log_diff = x - x.shift() 
    #    x_log_diff.dropna(inplace=True)
    #    plt.plot(results_AR.fittedvalues, color='red')
    #    plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-x_log_diff)**2))
    #    deseasonalized.plot()
        sarima_mod = SARIMAX(x, trend='n', order= order, seasonal_order= seasonal_order, enforce_stationarity=False).fit()
    #    print(sarima_mod.summary())
        forecast = sarima_mod.predict('2018-07-01', '2019-04-01')
        forecast.plot()
    #    D_data = x.diff().dropna()
    #    D_data.columns = [u'sales diff']
        
    #    D_data.plot()
        #y.rolling(6).std().plot()
        ax.legend(["2017-2018", "2019", "rolling", "predicted"])
        y.index=y.index.to_timestamp()
        a = y.corr(forecast)
        diff = y-forecast
        b = diff.std()
        c = "{:.2%}".format(abs(diff[0])/y[0])
        d = abs(y[0] - x[17])/abs(forecast[0] - x[17])
        dftest = adfuller(x, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
    #    print(dfoutput)
        
        dfoutput['CAT'] = cat
        dicky_test = dicky_test.append(dfoutput.transpose(),ignore_index=True)
        dicky_test['if_unitroot']=0
        dicky_test.loc[dicky_test['Critical Value (10%)']<dicky_test['Test Statistic'],'if_unitroot'] = 1
        dicky_test.loc[dicky_test.CAT == cat, 'AIC'] = sarima_mod.aic
        dicky_test.loc[dicky_test.CAT == cat, 'BIC'] = sarima_mod.bic
        dicky_test.loc[dicky_test.CAT == cat, 'corr'] = a
        dicky_test.loc[dicky_test.CAT == cat, 'diff_std'] = b
        dicky_test.loc[dicky_test.CAT == cat, 'diff_fore201901'] = c
        dicky_test.loc[dicky_test.CAT == cat, 'diff_18-19'] = d
        if (x.max()* (1+tol) < y.max()) or (x.min() > y.min() * (1+tol)):
            dicky_test.loc[dicky_test.CAT == cat, 'extremum'] =1
        else:
            dicky_test.loc[dicky_test.CAT == cat, 'extremum'] =0
        
        if (x.max()*(1+tol) < y.max()) or (x.min() > y.min() * (1+tol)) or (d>thresh):
            dicky_test.loc[dicky_test.CAT == cat, 'TAG'] = 1
            plt.savefig(outputs+ 'broken\\' + cat + '.png')
        else:
            dicky_test.loc[dicky_test.CAT == cat, 'TAG'] = 0
            plt.savefig(outputs + 'good\\'  + cat + '.png')
        
        plt.show()
    
    return dicky_test



import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def dtw_dist(x,y):
    dist, path = fastdtw(x,y,dist = euclidean)
    return dist

test_col = ['2019-01', '2019-02', '2019-03', '2019-04']
train_col =['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
       '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
       '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
       '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

test_col = ['2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12','2019-01', '2019-02', '2019-03', '2019-04']
train_col =['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
       '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
       '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06']




import datetime as dt

def TimeStamp_prep(df, keywrd = '_70K'):
    time_cols = [col for col in df.columns if keywrd in str(col)]
    time_cols.extend(['CATEGORY'])
    df_tmp = df[time_cols]
    df_tmp.columns = df_tmp.columns.str.replace(keywrd, '')
    df_tmp.set_index("CATEGORY", inplace=True)
    df_tmp.columns = df_tmp.columns.map(str)
    df_tmp.columns = df_tmp.columns.str.replace('14', '-')
    train_set = df_tmp[train_col]
    test_set = df_tmp[test_col]
    test_set = test_set.transpose()
    train_set = train_set.transpose()

    train_set.index = pd.to_datetime(train_set.index).to_period('m')
    train_set = train_set.sort_index()
    
    test_set.index = pd.to_datetime(test_set.index).to_period('m')
    test_set = test_set.sort_index()
        
    return train_set, test_set
    


