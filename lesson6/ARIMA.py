# -*- coding: utf-8 -*-
# 交通流量预测 ARIMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime, timedelta
import calendar

# 数据加载
train = pd.read_csv('train.csv')
# print(train.head)
# 转换为pandas中的日期格式
train['Datetime'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
# 将Datetime作为train的索引
train.index = train.Datetime
# print(train.head())
# 去掉ID，Datetime字段
train.drop(['ID', 'Datetime'], axis=1, inplace=True)
# print(train.head())
# 按照天进行采样
daily_train = train.resample('D').sum()
print(daily_train.head())

# daily_train作图
plt.figure(figsize=(30,7))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.plot(daily_train.Count, '-', label='按天')
plt.legend()
plt.show()

# 设置参数范围
ps = range(0, 15)
qs = range(0, 8)
ds = range(1, 4)
parameters = product(ps, ds, qs)
parameters_list = list(parameters)
# 寻找最优ARMA模型参数，即best_aic最小
results = []
best_aic = float("inf") # 正无穷
for param in parameters_list:
    try:
        # model = ARIMA(daily_train.Count,order=(param[0], param[1], param[2])).fit()
        # SARIMAX 包含季节趋势因素的ARIMA模型
        model = sm.tsa.statespace.SARIMAX(daily_train.Count,
                                order=(param[0], param[1], param[2]),
                                #seasonal_order=(4, 1, 2, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False).fit()

    except ValueError:
        print('参数错误:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
# 输出最优模型
print('最优模型: ', best_model.summary())

# 设置future_day，需要预测的时间date_list
daily_train2 = daily_train[['Count']]
future_day = 213
last_day = pd.to_datetime(daily_train2.index[len(daily_train2)-1])
print(last_day)
date_list = pd.date_range('2014-09-26',periods=future_day,freq='D')
print('date_list=', date_list)

# 添加未来要预测的7个月213天
future = pd.DataFrame(index=date_list, columns= daily_train.columns)
# print(future)
# print(len(daily_train2))
daily_train2 = pd.concat([daily_train2, future])
# print(daily_train2)

# 历史所有时间点+未来七个月进行预测，赋值到forecast字段里
daily_train2['forecast'] = best_model.predict(start=0, end=len(daily_train2))
# print(len(daily_train2))
# 第一个元素不正确，设置为NaN
daily_train2['forecast'][0] = np.NaN
print(daily_train2)

# 交通流量预测结果显示
plt.figure(figsize=(30,7))
daily_train2.Count.plot(label='实际流量')
daily_train2.forecast.plot(color='r', ls='--', label='预测流量')
plt.legend()
plt.title('交通流量预测')
plt.xlabel('日期')
plt.ylabel('流量')
plt.show()
