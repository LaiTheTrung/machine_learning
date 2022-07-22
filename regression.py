import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing,svm
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')
df= df[["Adj. Open","Adj. High","Adj. Low","Adj. Close","Adj. Volume" ]]
df['HL_pCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
df['pCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
df=df[['Adj. Open','Adj. Close','HL_pCT','pCT_change','Adj. Volume']]
forecast_col='Adj. Close'
df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-10)


print(df.head())