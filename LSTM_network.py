# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:20:24 2019

@author: seleneferro
"""
global output_dir
output_dir = 'C:\\Users\\base7005\\Documents\\TREND\\outputs\\'

raw_dir = 'C:\\Users\\base7005\\Documents\\TREND\\raw\\data_test\\'
data=pd.read_excel(raw_dir + 'pro_cat_ri.xlsx')  
print(data.head(5))  

data.replace(297169,858018,inplace=True)  
data.replace(570668,1227543,inplace=True)  

data.replace(111324,204853,inplace=True)  
data.replace(177306,403536,inplace=True)  
ts1=data[['Day','login']]  
ts2=data[['Day','delivery']]  
ts3=data[['Day','registration']]  
ts1.to_csv('ts1.csv',index=False)  
ts2.to_csv('ts2.csv',index=False)  
ts3.to_csv('ts3.csv',index=False)  
print(ts1.shape,ts2.shape,ts3.shape)


def timeseries_to_supervised(data,lag=1):  
    df = pd.DataFrame(data)  
    columns = [df.shift(i) for i in range(1,lag+1)]  
    columns.append(df)  
    df = pd.concat(columns,axis=1)  
    df.fillna(0,inplace=True)  
 return df  
 

def difference(dataset,interval=1):  
    diff =list()  
 for i in range(interval,len(dataset)):  
        value = dataset[i]-dataset[i-interval]  
        diff.append(value)  
 return pd.Series(diff)  
 
#差分逆变换
def inverse_difference(history,yhat,interval=1):  
 return yhat+history[-interval]  
 
#scale train and test data to [-1,1]  
def scale(train,test):  
    #fit scaler  
    scaler = MinMaxScaler(feature_range=(-1,1))  
    scaler = scaler.fit(train)  
    #transform train   
    train =train.reshape(train.shape[0],train.shape[1])  
    train_scaled = scaler.transform(train)  
    #transform test  
    test = test.reshape(test.shape[0],test.shape[1])  
    test_scaled = scaler.transform(test)  
 return scaler,train_scaled,test_scaled  
 

def invert_scale(scaler,X,value):  
    new_row = [x for x in X]+[value]  
    array = np.array(new_row)  
    array = array.reshape(1,len(array))  
    inverted = scaler.inverse_transform(array)  
 return inverted[0,-1]  
 
#LSTM
def fit_lstm(train, batch_size, nb_epoch, neurons):  
    X,y = train[:,0:-1], train[:, -1]  
    X = X.reshape(X.shape[0], 1,X.shape[1])  
 
    model = Sequential()  
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]),stateful=True))  
    model.add(Dense(1))  
    model.compile(loss='mean_squared_error',optimizer='adam')  
 
 for i in range(nb_epoch):  
        model.fit(X,y,epochs=1,batch_size=batch_size,verbose=0, shuffle=False)  
        model.reset_states()  
 return model  
 

def forecast_lstm(model, batch_size, X):  
    X = X.reshape(1,1,len(X))  
    yhat = model.predict(X, batch_size=batch_size)  
 return yhat[0,0]  
 
#定义数据预处理函数  
def data_processing(series):  
    raw_values = series.values  
    raw_values_log = np.log(raw_values)  
    diff_values = difference(raw_values_log,1)  
    supervised = timeseries_to_supervised(diff_values,1)  
    supervised_values = supervised.values  
 
      
    lastday=np.array([raw_values_log[-1],0]).reshape(1,2)  
    supervised_values_all = np.vstack((supervised_values,lastday))  
    train, test = supervised_values_all[0:584], supervised_values_all[584:]  
    scaler, train_scaled, test_scaled = scale(train,test)  
 
 return raw_values,raw_values_log,train_scaled, test_scaled,scaler  
 
""" 
forecast func
"""  
def  model_fit_pred(train_scaled,test_scaled,nb_echop,scaler,raw_values_log):  
 
    nb_echop=nb_echop  
    scaler=scaler  
    raw_values_log=raw_values_log  
   
    start = time.time()  
    lstm_model = fit_lstm(train_scaled, 1, nb_echop, 1)  
 print(">--------- Compilation Time : ", time.time() - start)  
 
    train_reshaped = train_scaled[:,0].reshape(len(train_scaled),1,1)  
    lstm_model.predict(train_reshaped,batch_size=1)  
 
    predictions = list()  
 for i in range(len(test_scaled)):  

        X,y = test_scaled[i,0:-1], test_scaled[i,-1]  
        yhat = forecast_lstm(lstm_model,1,X)  

        yhat = invert_scale(scaler,X,yhat)  
        yhat = inverse_difference(raw_values_log,yhat,len(test_scaled)+1-i)  
        yhat = np.exp(yhat)  
        predictions.append(yhat)  
 
 return predictions,lstm_model  
 

def model_performance(raw_values,predictions):  
    rmse = np.sqrt(mean_squared_error(raw_values[583:-2],predictions[:-1]))  
    rcParams['figure.figsize']=15,6  
    plt.plot(raw_values[583:-2])  
    plt.plot(predictions[::])  
    plt.title("the performance RMSE of the LSTM  on the test data is %.5f"%rmse)  
 print("the forcast number of tomorrow is %.6f"%predictions[-1])  
 print("误差百分比是 %.6f"%(rmse/np.mean(raw_values[583:-2])))  
    plt.savefig('performance.png')  


#load dataset  
parser = lambda dates: datetime.strptime(dates,'%Y-%m-%d')  
series1 = pd.read_csv('ts1.csv',header=0,parse_dates=['Day'], index_col='Day', squeeze=True, date_parser=parser)  
series2 = pd.read_csv('ts2.csv',header=0,parse_dates=['Day'], index_col='Day', squeeze=True, date_parser=parser)  
series3 = pd.read_csv('ts3.csv',header=0,parse_dates=['Day'], index_col='Day', squeeze=True, date_parser=parser)  
 
raw_values1,raw_values_log1,train_scaled1, test_scaled1,scaler1 =data_processing(series1)  
raw_values2,raw_values_log2,train_scaled2, test_scaled2,scaler2 =data_processing(series2)  
raw_values3,raw_values_log3,train_scaled3, test_scaled3,scaler3 =data_processing(series3)  
 

predictions1,lstm_model1 = model_fit_pred(train_scaled1,test_scaled1,100,scaler1,raw_values_log1)  
predictions2,lstm_model2 = model_fit_pred(train_scaled2,test_scaled2,100,scaler2,raw_values_log2)  
predictions3,lstm_model3 = model_fit_pred(train_scaled3,test_scaled3,100,scaler3,raw_values_log3) 
