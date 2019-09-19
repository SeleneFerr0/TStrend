# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:49:38 2019

@author: SELENEFERRO
"""


from __future__ import print_function

import xlrd
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
#from bokeh.io import output_file, show
#from bokeh.plotting import figure
#from numpy import histogram, linspace
import pytz
from datetime import datetime, timedelta
from pandas import ExcelWriter
import math
import numpy as np
from scipy import stats

import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf  #autocorrelation
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox #white noise
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
#%matplotlib inline 
#
#%pylab inline



sys.path.append(r'C:\Users\base7005\Documents\TREND\scripts')
from TimeSeries_fun_01 import *



my_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d-%H")
# =============================================================================
# Import Data
# =============================================================================
dir_work = r'C:\Users\base7005\Documents\TREND\raw\data_test'
#os.chdir(dir_work)
#files_all = os.listdir(dir_work)
#files_all.sort()



global output_dir
output_dir = 'C:\\Users\\base7005\\Documents\\TREND\\outputs\\'

raw_dir = 'C:\\Users\\base7005\\Documents\\TREND\\raw\\data_test\\'
#df_raw_val = pd.DataFrame(pd.read_excel(raw_dir + 'impact_pro_ub_type_cat.xlsm', sheet_name="val",encoding='gb18030'))
#df_raw_vol = pd.DataFrame(pd.read_excel(raw_dir + 'impact_pro_ub_type_cat.xlsm', sheet_name="vol",encoding='gb18030'))

df_raw_val = pd.DataFrame(pd.read_excel(raw_dir + 'pro_cat_ri.xlsm', sheet_name="val",encoding='gb18030'))


#shoptype = 'sum smkt'
shoptype ='nssg'
rup_point = '2018-07'

bau_cols = [col for col in df_raw_val.columns if '_CELLIST' in str(col)]
base_cols = [col for col in df_raw_val.columns if '_1' in str(col)]
cce_cols = [col for col in df_raw_val.columns if '_2' in str(col)]


#df_raw_tmp = df_raw_val[df_raw_val.SHOPTYPE == shoptype]
#df_raw_tmp.drop("SHOPTYPE", axis =1, inplace=True)

df_raw_tmp = df_raw_val.copy()
#df_raw_tmp.rename(columns={'CATEGORYCODE':'CATEGORY'}, inplace=True)
df_raw_tmp['CATEGORY'] = df_raw_tmp['CATEGORYCODE'] + '_' + df_raw_tmp['PROVINCE']
df_raw_tmp['shoptype'] = 'all'
train_cce, test_cce = TimeStamp_prep(df_raw_tmp, keywrd = '_1')


thresh = 1.5
tol = 0.5

categories = list(train_cce.columns)
#cat = categories[0]

#cat=categories[0]
shoptype = 'all'
dicky_test = sarima_detect(train_cce, test_cce, shoptype, categories)

writer = ExcelWriter('C:\\Users\\base7005\\Documents\\TREND\\outputs\\' +  'ADF_test_' + shoptype +'.xlsx')
dicky_test.to_excel(writer,'0912',index=False)
writer.save()




#cce_col = list(set(df_raw_val.columns) - set(bau_cols) - set(['PROVINCE', 'URBAN_RURAL', 'SHOPTYPE', 'CATEGORY']))
test_col = ['2019-01', '2019-02', '2019-03', '2019-04']
train_col =['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
       '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
       '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
       '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

df_val_urban = df_raw_val[df_raw_val.URBAN_RURAL =='Urban']
df_val_urban = df_val_urban.drop(bau_cols, axis=1)
df_val_urban = df_val_urban.groupby(['SHOPTYPE','CATEGORY']).sum().reset_index()


df_val_rural = df_raw_val[df_raw_val.URBAN_RURAL =='Rural']
df_val_rural = df_val_rural.drop(bau_cols, axis=1)
df_val_rural = df_val_rural.groupby(['SHOPTYPE','CATEGORY']).sum().reset_index()



'''loop start by shoptype'''



shoptypes= list(df_val_urban.SHOPTYPE.unique())

shoptype = 'sum smkt'
rup_point = '2019-01'


df_tmp_val = df_val_urban[df_val_urban.SHOPTYPE == shoptype]
df_tmp_val.set_index("CATEGORY", inplace=True)
df_tmp_val = df_tmp_val[cce_col]
#df_tmp_vol.set_index("CATEGORY", inplace=True)
df_tmp_val.columns = df_tmp_val.columns.map(str)
#df_tmp_vol.columns = df_tmp_vol.columns.map(str)
df_tmp_val.columns = df_tmp_val.columns.str.replace('14', '-')
#df_tmp_vol.columns = df_tmp_vol.columns.str.replace('14', '-')


train_set = df_tmp_val[train_col]
test_set = df_tmp_val[test_col]
#test_set.rename(columns=lambda x: datetime(year=int(str(x)[0:4]), month=int(str(x)[6:8]), day=28), inplace=True)


'''BAU line'''


bau_val_urban = df_raw_val[(df_raw_val.URBAN_RURAL =='Urban') &(df_raw_val.SHOPTYPE == shoptype)]
bau_val_urban = bau_val_urban.groupby(['SHOPTYPE','CATEGORY']).sum().reset_index()

bau_cols.extend(['CATEGORY'])
bau_val_urban = bau_val_urban[bau_cols]
bau_val_urban.columns = bau_val_urban.columns.str.replace('70K_', '')
bau_val_urban.set_index("CATEGORY", inplace=True)

#df_tmp_vol.set_index("CATEGORY", inplace=True)
bau_val_urban.columns = bau_val_urban.columns.map(str)
#df_tmp_vol.columns = df_tmp_vol.columns.map(str)
bau_val_urban.columns = bau_val_urban.columns.str.replace('14', '-')




test_set = test_set.transpose()
train_set = train_set.transpose()
import datetime as dt
train_set.index = pd.to_datetime(train_set.index).to_period('m')
train_set = train_set.sort_index()

test_set.index = pd.to_datetime(test_set.index).to_period('m')
test_set = test_set.sort_index()




'''GARCH'''
cat='ATD'
x = np.log(train_set[cat]+1)
y = np.log(test_set[cat]+1)


from arch import arch_model
'''pip install arch --install-option="--no-binary"'''
'''not suitable for these sets

garch_mod = arch_model(x, mean='Zero', vol='ARCH', p=12,q=12)
fit1 = garch_mod.fit()
print(fit1.summary())

'''




##########################################################################################################
from datetime import date, timedelta
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)

df_test = pd.read_csv(
    "../input/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

items = pd.read_csv(
    "../input/items.csv",
).set_index("item_nbr")

stores = pd.read_csv(
    "../input/stores.csv",
).set_index("store_nbr")

le = LabelEncoder()
items['family'] = le.fit_transform(items['family'].values)

stores['city'] = le.fit_transform(stores['city'].values)
stores['state'] = le.fit_transform(stores['state'].values)
stores['type'] = le.fit_transform(stores['type'].values)

df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]
del df_train

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

items = items.reindex(df_2017.index.get_level_values(1))
stores = stores.reindex(df_2017.index.get_level_values(0))


df_2017_item = df_2017.groupby('item_nbr')[df_2017.columns].sum()
promo_2017_item = promo_2017.groupby('item_nbr')[promo_2017.columns].sum()

df_2017_store_class = df_2017.reset_index()
df_2017_store_class['class'] = items['class'].values
df_2017_store_class_index = df_2017_store_class[['class', 'store_nbr']]
df_2017_store_class = df_2017_store_class.groupby(['class', 'store_nbr'])[df_2017.columns].sum()

df_2017_promo_store_class = promo_2017.reset_index()
df_2017_promo_store_class['class'] = items['class'].values
df_2017_promo_store_class_index = df_2017_promo_store_class[['class', 'store_nbr']]
df_2017_promo_store_class = df_2017_promo_store_class.groupby(['class', 'store_nbr'])[promo_2017.columns].sum()

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(df, promo_df, t2017, is_train=True, name_prefix=None):
    X = {
        "promo_14_2017": get_timespan(promo_df, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_df, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_df, t2017, 140, 140).sum(axis=1).values,
        "promo_3_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 3).sum(axis=1).values,
        "promo_7_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 7).sum(axis=1).values,
        "promo_14_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 14).sum(axis=1).values,
    }

    for i in [3, 7, 14, 30, 60, 140]:
        tmp1 = get_timespan(df, t2017, i, i)
        tmp2 = (get_timespan(promo_df, t2017, i, i) > 0) * 1

        X['has_promo_mean_%s' % i] = (tmp1 * tmp2.replace(0, np.nan)).mean(axis=1).values
        X['has_promo_mean_%s_decay' % i] = (tmp1 * tmp2.replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

        X['no_promo_mean_%s' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan)).mean(axis=1).values
        X['no_promo_mean_%s_decay' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

    for i in [3, 7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t2017, i, i)
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s' % i] = tmp.mean(axis=1).values
        X['median_%s' % i] = tmp.median(axis=1).values
        X['min_%s' % i] = tmp.min(axis=1).values
        X['max_%s' % i] = tmp.max(axis=1).values
        X['std_%s' % i] = tmp.std(axis=1).values

    for i in [3, 7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t2017 + timedelta(days=-7), i, i)
        X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s_2' % i] = tmp.mean(axis=1).values
        X['median_%s_2' % i] = tmp.median(axis=1).values
        X['min_%s_2' % i] = tmp.min(axis=1).values
        X['max_%s_2' % i] = tmp.max(axis=1).values
        X['std_%s_2' % i] = tmp.std(axis=1).values

    for i in [7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t2017, i, i)
        X['has_sales_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
        X['last_has_sales_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
        X['first_has_sales_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

        tmp = get_timespan(promo_df, t2017, i, i)
        X['has_promo_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
        X['last_has_promo_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
        X['first_has_promo_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

    tmp = get_timespan(promo_df, t2017 + timedelta(days=16), 15, 15)
    X['has_promo_days_in_after_15_days'] = (tmp > 0).sum(axis=1).values
    X['last_has_promo_day_in_after_15_days'] = i - ((tmp > 0) * np.arange(15)).max(axis=1).values
    X['first_has_promo_day_in_after_15_days'] = ((tmp > 0) * np.arange(15, 0, -1)).max(axis=1).values

    for i in range(1, 16):
        X['day_%s_2017' % i] = get_timespan(df, t2017, i, 1).values.ravel()

    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df, t2017, 140-i, 20, freq='7D').mean(axis=1).values

    for i in range(-16, 16):
        X["promo_{}".format(i)] = promo_df[t2017 + timedelta(days=i)].values.astype(np.uint8)

    X = pd.DataFrame(X)

    if is_train:
        y = df[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X

print("Preparing dataset...")
t2017 = date(2017, 6, 14)
num_days = 6
X_l, y_l = [], []
for i in range(num_days):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(df_2017, promo_2017, t2017 + delta)

    X_tmp2 = prepare_dataset(df_2017_item, promo_2017_item, t2017 + delta, is_train=False, name_prefix='item')
    X_tmp2.index = df_2017_item.index
    X_tmp2 = X_tmp2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

    X_tmp3 = prepare_dataset(df_2017_store_class, df_2017_promo_store_class, t2017 + delta, is_train=False, name_prefix='store_class')
    X_tmp3.index = df_2017_store_class.index
    X_tmp3 = X_tmp3.reindex(df_2017_store_class_index).reset_index(drop=True)

    X_tmp = pd.concat([X_tmp, X_tmp2, X_tmp3, items.reset_index(), stores.reset_index()], axis=1)
    X_l.append(X_tmp)
    y_l.append(y_tmp)

    del X_tmp2
    gc.collect()

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)

del X_l, y_l
X_val, y_val = prepare_dataset(df_2017, promo_2017, date(2017, 7, 26))

X_val2 = prepare_dataset(df_2017_item, promo_2017_item, date(2017, 7, 26), is_train=False, name_prefix='item')
X_val2.index = df_2017_item.index
X_val2 = X_val2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

X_val3 = prepare_dataset(df_2017_store_class, df_2017_promo_store_class, date(2017, 7, 26), is_train=False, name_prefix='store_class')
X_val3.index = df_2017_store_class.index
X_val3 = X_val3.reindex(df_2017_store_class_index).reset_index(drop=True)

X_val = pd.concat([X_val, X_val2, X_val3, items.reset_index(), stores.reset_index()], axis=1)

X_test = prepare_dataset(df_2017, promo_2017, date(2017, 8, 16), is_train=False)

X_test2 = prepare_dataset(df_2017_item, promo_2017_item, date(2017, 8, 16), is_train=False, name_prefix='item')
X_test2.index = df_2017_item.index
X_test2 = X_test2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

X_test3 = prepare_dataset(df_2017_store_class, df_2017_promo_store_class, date(2017, 8, 16), is_train=False, name_prefix='store_class')
X_test3.index = df_2017_store_class.index
X_test3 = X_test3.reindex(df_2017_store_class_index).reset_index(drop=True)

X_test = pd.concat([X_test, X_test2, X_test3, items.reset_index(), stores.reset_index()], axis=1)

del X_test2, X_val2, df_2017_item, promo_2017_item, df_2017_store_class, df_2017_promo_store_class, df_2017_store_class_index
gc.collect()

print("Training and predicting models...")
params = {
    'num_leaves': 80,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 16
}

MAX_ROUNDS = 5000
val_pred = []
test_pred = []
cate_vars = []
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items["perishable"]] * num_days) * 0.25 + 1
    )
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=items["perishable"] * 0.25 + 1,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=125, verbose_eval=50
    )
    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))
    val_pred.append(bst.predict(
        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

print("Validation mse:", mean_squared_error(
    y_val, np.array(val_pred).transpose()))

weight = items["perishable"] * 0.25 + 1
err = (y_val - np.array(val_pred).transpose())**2
err = err.sum(axis=1) * weight
err = np.sqrt(err.sum() / weight.sum() / 16)
print('nwrmsle = {}'.format(err))

y_val = np.array(val_pred).transpose()
df_preds = pd.DataFrame(
    y_val, index=df_2017.index,
    columns=pd.date_range("2017-07-26", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
df_preds["unit_sales"] = np.clip(np.expm1(df_preds["unit_sales"]), 0, 1000)
df_preds.reset_index().to_csv('lgb_cv.csv', index=False)

print("Making submission...")
y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('lgb_sub.csv', float_format='%.4f', index=None)







##########################################################################################################
#D_data = y.diff().dropna()
#D_data.columns = [u'sales diff']
#D_data.plot()
#
#
#
#sarima_mod = sm.tsa.statespace.SARIMAX(x, trend='n', order=(0,0,0), seasonal_order=(0, 1, 0, 12),enforce_stationarity=False).fit()
#print(sarima_mod.summary())
#forecast = sarima_mod.predict('2019-01-01', '2019-04-01')
#plt.plot(forecast)
#plt.show()
#
#
#plt.plot(x_log_diff,color='green')
#x_log_diff.dropna(inplace=True)
#plt.plot(results_AR.fittedvalues, color='red')
#
#
##result_mul = seasonal_decompose(x, model='multiplicative')
#result_add = seasonal_decompose(x, model='additive')
#
#x.interpolate(inplace = True)
#x.index=x.index.to_timestamp()
#result_mul = seasonal_decompose(x)
#deseasonalized = x / result_mul.seasonal
#
#
#x_log_diff = x - x.shift()
#x_log_diff.plot()
#
#
#
#model = ARIMA(x, order=(6, 1, 1)) 
#results_AR = model.fit(disp=0,transparams=True)  
#plt.plot(x_log_diff,color='green')
#x_log_diff.dropna(inplace=True)
#plt.plot(results_AR.fittedvalues, color='red')
##plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-x_log_diff)**2))
##forecast = results_AR.forecast()[0]
#
#forecast = results_AR.predict('2019-01-01', '2019-04-01')
#forecast = inverse_difference(x, forecast, 24)
#plt.plot(forecast)
#plt.show()








#import pandas as pd
#import numpy as np
#import lightgbm as lgb
#from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import LabelEncoder
#import gc
#
#
#def nwrmsle(yval, ypred, weights=None):
#    return np.sqrt(mean_squared_error(np.log(1+yval), np.log(1+ypred), sample_weight=weights))
#
#
#def combine_lags(df0, y0=None):
#
#    test_range = pd.date_range('2017-08-16', '2017-08-31')
#
#    features_arr = []
#    y_arr = []
#
#    for i in range(16):
#
#        if y0 is not None:
#            date = pd.to_datetime(y0.columns[i])
#        else:
#            date = test_range[i]
#        day = date.day
#        dow = date.dayofweek
#
#        df = df0[['mean_1', 'mean_3', 'mean_7', 'mean_14', 'mean_28', 'mean_60', 'mean_90', 'mean_180', 'mean_365',
#                       'mean_diff_7_28', 'mean_diff_14_60', 'mean_diff_28_90',
#                       'mean_zerodays_7', 'mean_zerodays_28', 'mean_zerodays_90', 'mean_zerodays_365',
#                       'mean_zerodays_diff_7_28', 'mean_zerodays_diff_28_90',
#                       'promo_mean_28', 'promo_mean_365', 'promo_mean_7', 'promo_mean_90',
#                       'mean_store_family_7', 'mean_store_family_28', 'mean_store_family_diff_7_28',
#                       'mean_store_class_7', 'mean_store_class_28', 'mean_store_class_diff_7_28',
#                       'mean_item_city_7', 'mean_item_city_28', 'mean_item_city_diff_7_28',
#                       'mean_item_state_7', 'mean_item_state_28', 'mean_item_state_diff_7_28',
#                       'mean_item_type_7', 'mean_item_type_28', 'mean_item_type_diff_7_28',
#                       'mean_item_cluster_7', 'mean_item_cluster_28', 'mean_item_cluster_diff_7_28',
#                       'mean_item_7', 'mean_item_28', 'mean_item_diff_7_28',
#                       'mean_store_zerodays_7', 'mean_store_zerodays_28', 'mean_store_zerodays_diff_7_28']]
#
#        df['mean_dow_7'] = df0['mean_7_dow%d' % dow]
#        df['mean_dow_14'] = df0['mean_14_dow%d' % dow]
#        df['mean_dow_28'] = df0['mean_28_dow%d' % dow]
#        df['mean_dow_56'] = df0['mean_56_dow%d' % dow]
#        df['mean_dow_84'] = df0['mean_84_dow%d' % dow]
#        df['mean_dow_168'] = df0['mean_168_dow%d' % dow]
#        df['mean_dow_364'] = df0['mean_364_dow%d' % dow]
#
#        df['mean_day_365'] = df0['mean_365_day%d' % day]
#
#        df['regression_50'] = df0['regression50_%d' % i]
#        df['regression_100'] = df0['regression100_%d' % i]
#
#        df['promo'] = df0['promo_%d' % i]
#        df['promo_mean'] = df0[['promo_0', 'promo_1', 'promo_2', 'promo_3', 'promo_4', 'promo_5', 'promo_6', 'promo_7',
#                'promo_8', 'promo_9', 'promo_10', 'promo_11', 'promo_12', 'promo_13', 'promo_14', 'promo_15']].mean(axis=1)
#
#        df['family'] = items2['family'].values
#        df['class'] = items2['class'].values
#        df['perishable'] = items2['perishable'].values
#        df['city'] = stores2['city'].values
#        df['state'] = stores2['state'].values
#        df['type'] = stores2['type'].values
#        df['cluster'] = stores2['cluster'].values
#
#        df['dow'] = dow
#        df['day'] = day
#
#        df = df.reset_index()
#        df['date'] = date
#        df['lag'] = i
#
#        features_arr.append(df)
#
#        if y0 is not None:
#            y_i = y0.iloc[:,i].rename('y').to_frame()
#            y_i['date'] = date
#            y_i = y_i.reset_index().set_index(['store_nbr', 'item_nbr', 'date']).squeeze()
#            y_arr.append(y_i)
#
#    features  = pd.concat(features_arr)
#    if y0 is not None:
#        y  = pd.concat(y_arr)
#        del features_arr, y_arr
#        return features, y
#    else:
#        del features_arr
#        return features
#
#
#
## prepare data
#
#val0 = pd.read_csv('features/val_data.csv', index_col=[0,1])
#test0 = pd.read_csv('features/test_data.csv', index_col=[0,1])
#yval0 = pd.read_csv('target/y_val.csv', index_col=[0,1])
#
#t = pd.to_datetime('2017-07-05')
#train_start = []
#for i in range(25):
#    delta = pd.Timedelta(days=7 * i)
#    train_start.append((t-delta).strftime('%Y-%m-%d'))
#print(train_start)
#
#train0 = []
#ytrain0 = []
#for start in train_start:
#    print(start)
#    train0.append(pd.read_csv('features/train_data_%s.csv' % start, index_col=[0,1]))
#    ytrain0.append(pd.read_csv('target/y_train_%s.csv' % start, index_col=[0,1]))
#
#
#items = pd.read_csv('data/items.csv')
#stores = pd.read_csv('data/stores.csv')
#
#le = LabelEncoder()
#items.family = le.fit_transform(items.family)
#stores.city = le.fit_transform(stores.city)
#stores.state = le.fit_transform(stores.state)
#stores.type = le.fit_transform(stores.type)
#
#items2 = items.set_index('item_nbr').reindex(val0.index.get_level_values(1))
#print(items2.shape)
#stores2 = stores.set_index('store_nbr').reindex(val0.index.get_level_values(0))
#print(stores2.shape)
#
#
#val, yval = combine_lags(val0, yval0)
#test = combine_lags(test0)
#
#train = []
#ytrain = []
#for i in range(len(train0)):
#    tr, ytr = combine_lags(train0[i], ytrain0[i])
#    train.append(tr)
#    ytrain.append(ytr)
#train = pd.concat(train)
#ytrain = pd.concat(ytrain)
#print(train.shape, ytrain.shape)
#
#train.drop('date', axis=1, inplace=True)
#val.drop('date', axis=1, inplace=True)
#test2 = test.drop('date', axis=1)
#
#gc.collect()
#
#
#
## train model
#
#clf = lgb.LGBMRegressor(n_estimators=5000, learning_rate=0.05, num_leaves=150, min_data_in_leaf=200,
#                        subsample=0.7, colsample_bytree=0.3, random_state=42, n_jobs=-1)
#
#clf.fit(train, np.log(1+ytrain), eval_set=[(val, np.log(1+yval))], early_stopping_rounds=50,
#                eval_metric='rmse', sample_weight=(train.perishable*0.25+1).values, verbose=20)
#pred = np.exp(clf.predict(val, num_iteration=clf.best_iteration_))-1
#test_pred = np.exp(clf.predict(test2, num_iteration=clf.best_iteration_))-1
#
#print('rmsle %.5f' % nwrmsle(yval.values, pred))
#print('nwrmsle %.5f' % nwrmsle(yval.values, pred, weights=val.perishable.values*0.25+1))
#val_range = np.sort(yval.index.get_level_values(2).unique())
#first5_idx = yval.index.get_level_values(2).isin(val_range[:5])
#last11_idx = yval.index.get_level_values(2).isin(val_range[5:])
#print('nwrmsle-first5 %.5f' % nwrmsle(yval.values[first5_idx], pred[first5_idx],
#                                      weights=val.perishable.values[first5_idx]*0.25+1))
#print('nwrmsle-last11 %.5f' % nwrmsle(yval.values[last11_idx], pred[last11_idx],
#                                      weights=val.perishable.values[last11_idx]*0.25+1))
#
#
#
## make submission
#
#df_test = pd.read_csv("data/test.csv", usecols=[0, 1, 2, 3], parse_dates=["date"])
#                    set_index(['store_nbr', 'item_nbr', 'date'])
#test_pred2 = pd.Series(test_pred, index=test.set_index(['store_nbr', 'item_nbr', 'date']).index, name='unit_sales')
#submission = df_test.join(test_pred2, how="left").fillna(0)
#submission.unit_sales = submission.unit_sales.clip(lower=0)
#
#submission.to_csv('submissions/submission.csv', float_format='%.6f', index=False)
#
#
#


