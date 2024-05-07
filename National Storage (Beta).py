#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from xgboost import *

# model = XGBRegressor()


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skewnorm, yeojohnson, boxcox, zscore
from statsmodels.tsa.stattools import adfuller 
import tslearn
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import r2_score, mean_squared_error

# Scale variables
from sklearn.preprocessing import scale, StandardScaler
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from sklearn.preprocessing import robust_scale, RobustScaler

# Transform variables
from sklearn.preprocessing import quantile_transform, QuantileTransformer
from sklearn.preprocessing import power_transform, PowerTransformer

# Encode categorical variables
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

# Discretize continuous variables
from sklearn.preprocessing import KBinsDiscretizer

# Impute missing values
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

ns = pd.read_csv("P:/KJ/Coding Projects/Storage Regression (R Project)/national4.csv", index_col = 0, header = 0)
ns.index = pd.DatetimeIndex(ns.index.values, freq = 'infer')
ns.sort_index(ascending = True, inplace = True)
# ns.Date = pd.to_datetime(ns.Date).dt.date
# ns.sort_values('Date', inplace = True)
# ns.reset_index(drop = True, inplace = True)

del ns['123086']
del ns['408920']
del ns['146847']
del ns['89088']
del ns['146904']

# ns = ns.diff().dropna()
ns


# In[2]:


#Apply first differencing to the data
# ns = ns.diff().dropna()
# display(ns)
#Create the dataset matrices
X = ns.drop(labels = ['EIA'], axis = 1)
y = ns.EIA
# y2 = ns.iloc[:,0]
# y2

# X = ns.iloc[:,1:]
# y= ns.iloc[:, :0]

# X_train = ns.iloc[:-10, 1:].values
# X_test = ns.iloc[-10:, 1:].values
# y_train = ns.iloc[:-10,:0].values
# y_test = ns.iloc[-10:,:0].values


# In[3]:


# y_train.shape
X.shift().dropna()
y.shift().dropna()
y
# ?TimeSeriesSplit


# In[4]:


#Split into train | test
from sklearn.model_selection import TimeSeriesSplit
tss = TimeSeriesSplit(n_splits = 5)
for train_index, test_index in tss.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# In[5]:


# y_train.values.reshape(-1,1).shape
X_train.mean(axis = 0)
# g1 = X_sc.fit_transform()


# In[6]:


#Plot the target split
y_train.groupby(y_train.index).mean().plot()
y_test.groupby(y_test.index).mean().plot(title = 'EIA')
plt.show()


# In[7]:


#Feature Scaling
X_sc = MinMaxScaler()
y_sc = MinMaxScaler()

X_train = X_sc.fit_transform(X_train.values)
y_train = y_sc.fit_transform(y_train.values.reshape(-1,1))
# y_train = y_train.flatten()

# X_train = X_train.diff().dropna()

# for ea_col, data in X_train.iteritems():
#     display(adf_test(X_train[ea_col]))


# In[8]:


# X
y2 = np.array(y).reshape(-1,1)
type(y2)
# display(y2.shape)
display(y_train.shape)


# In[9]:


# y_train = y_train.ravel().reshape(1,-1).shape
# B = np.reshape(y_train, (-1, 1))
# B
# X_train.reshape(len(X.columns),-1).shape
print(y_train.ravel().shape)
X_train.reshape(-1, len(X.columns)).shape


# In[10]:


#This cell will be deprecated following the general SVR method

# from tslearn.svm import TimeSeriesSVR
# reg = TimeSeriesSVR(kernel = 'rbf')
# # reg.fit(X_train, y_train).predict(X_test).shape
# y_train.shape


# In[11]:


#Train the model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
# regressor = SVR()
## regressor.fit(X_train.reshape(len(X.columns),-1), y_train.ravel())
# regressor.fit(X_train, y_train.flatten())

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'sigmoid', 'poly', 'linear'],
              'epsilon' : [0.1, 0.05, 0.01, 0.005, 0.001]} 
  
grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3)

grid.fit(X_train, y_train.flatten())


# In[12]:


# X_sc.transform(X_test).reshape(-1, len(X.columns)).shape
print(grid.best_params_)
print("")
print(grid.best_estimator_)


# In[13]:


#Predict the result
y_pred = grid.predict(X_sc.transform(X_test.values))
y_pred = y_sc.inverse_transform(y_pred.reshape(-1,1))


# In[14]:


# regressor.predict(X_sc.transform(X_test.values))
# y_pred


# In[15]:


# y_test = y_test
df = pd.DataFrame({'Predicted Value' : y_pred.flatten(), 'Real Value' : y_test, 'Error' : y_pred.flatten() - y_test})
display(df)


print('R2: {}'.format(r2_score(y_test, y_pred)))
print('RMSE: {}'.format(mean_squared_error(y_test, y_pred.flatten(), squared=False)))

plt.plot(df.iloc[:,0], label = 'Prediction')
plt.plot(df.iloc[:,1], label = "Target")
plt.plot(df.iloc[:,2], label = 'Model Loss')
plt.axhline(linestyle = '--', color = 'red')
plt.legend()
plt.show()


# In[16]:


import pickle

filename = 'P:/KJ/Coding Projects/Storage Regression (R Project)/nick.pkl'
pickle.dump(grid, open(filename, 'wb'))


# In[17]:


unseen = pd.read_excel(r'P:\KJ\Coding Projects\Storage Regression (R Project)\unseen_data.xlsx', header = 2, index_col = 0, usecols = "C:EN")
unseen.dropna(axis = 0, inplace = True)


# In[18]:


unseen_final = unseen.iloc[11:,:].copy()
unseen_final.index = pd.DatetimeIndex(unseen_final.index.values, freq = 'infer')
unseen_final.sort_index(ascending = True, inplace = True)

del unseen_final[123086]
del unseen_final[408920]
del unseen_final[146847]
del unseen_final[89088]
del unseen_final[146904]

unseen_final


# In[19]:


pickled_model = pickle.load(open(filename, 'rb'))
result = pickled_model.predict(X_sc.transform(unseen_final.values))
result = y_sc.inverse_transform(result.reshape(-1,1))


# In[21]:


# a = input("Please enter this week's expected storage value: ")
y_test = pd.Series([111,52,107,79,64,-50])
df = pd.DataFrame({'Predicted Value' : result.flatten(), 'Real Value' : y_test, 'Error' : result.flatten() - y_test})
df.index = unseen_final.index
display(df)


print('R2: {}'.format(r2_score(y_test, result)))
print('RMSE: {}'.format(mean_squared_error(y_test, result.flatten(), squared=False)))

plt.plot(df.iloc[:,0], label = 'Prediction')
plt.plot(df.iloc[:,1], label = "Target")
plt.plot(df.iloc[:,2], label = 'Model Loss')
plt.axhline(linestyle = '--', color = 'red')
plt.xticks(rotation = 45) 
plt.legend()
plt.show()


# #### End of Nick Model

# In[21]:


# counter = 0 
for ea_col, all_rows in ns.iteritems():
#     counter += 1
    plt.plot(ns[ea_col])
    plt.title(ea_col)
#     plt.subplots()
    plt.show()
#     if counter == 3:
#         break


# In[22]:


df_differenced['146904']


# In[ ]:


for ea_col, all_rows in df_differenced.iteritems():
#     counter += 1
    plt.plot(df_differenced[ea_col])
    plt.title(ea_col)
#     plt.subplots()
    plt.show()


# In[ ]:


def highlight_cells(val):
    color = 'red' if val == 0 else ''
    return 'background-color: {}'.format(color)

pd.options.display.max_rows = 142
ns.describe(include = 'all').T.style.applymap(highlight_cells)


# In[ ]:


values = ns.values

parts = len(values) // 3 

part_1, part_2, part_3 = values[:parts], values[parts:(parts*2)], values[(parts*2):(parts*3)]

print("", int(np.mean(part_1)), '\t', np.var(part_1), '\n', int(np.mean(part_2)), '\t', np.var(part_2), '\n', int(np.mean(part_3)), '\t', np.var(part_3))


# In[ ]:


del ns['123086']
del ns['408920']
del ns['146847']

values = np.log(ns.values)

print(values[:15])

part_1, part_2, part_3 = values[:parts], values[parts:(parts*2)], values[(parts*2):(parts*3)]

print("", np.mean(part_1), '\t', np.var(part_1), '\n', np.mean(part_2), '\t', np.var(part_2), '\n', np.mean(part_3), '\t', np.var(part_3))

plt.plot(values)


# In[ ]:


values = ns.EIA.values 

res = adfuller(values)

res


# In[ ]:


pd.options.display.max_columns = 142
cor_matrix = ns.corr().abs()

upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
upper_tri


# In[ ]:


to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(); print(to_drop)

ns1 = ns.drop(columns = to_drop, axis=1)
print(); display(ns1.head(10))


# In[ ]:


scaler = StandardScaler(with_std=False)

ns_centered = scaler.fit_transform(ns.values)
print('mean (original): {}    mean (centered): {}'.format(round(ns.mean()), round(ns_centered.mean())))


# In[ ]:


import seaborn as sns
fig, axes = plt.subplots(figsize=(12, 5), ncols=2)
sns.distplot(ns,          ax=axes[0], kde=False, rug=False).set_title('Original')
sns.distplot(ns_centered, ax=axes[1], kde=False, rug=False).set_title('Centered')


# In[ ]:


scaler = StandardScaler()

ns_scaled = scaler.fit_transform(ns.values)#.reshape(-1, 1))
print('mean (original): {}  mean (scaled): {}  std (original): {}  std (scaled): {}'.format(round(ns.values.mean()), 
                                             abs(round(ns_scaled.mean())),
                                             round(ns.values.std()),
                                             round(ns_scaled.std())
                                            ))


# In[ ]:


fig, axes = plt.subplots(figsize=(12, 5), ncols=2)
sns.distplot(ns.values,        ax=axes[0], kde=False, rug=False, fit = stats.norm).set_title('Original')
sns.distplot(ns_scaled, ax=axes[1], kde=False, rug=False, fit = stats.norm).set_title('Centered and Scaled');


# In[ ]:


import statsmodels.api as sm
from statsmodels.tsa.api import VAR

model = VAR(ns1)
model_fit = model.fit()


pred = model_fit.forecast(model_fit.endog, steps=1)
print(pred)


# In[ ]:


nobs = 15
df_train, df_test = ns1[0:-nobs], ns1[-nobs:]
df_train


# In[ ]:


# Augmented Dickey-Fuller Test (ADF Test)/unit root test
from statsmodels.tsa.stattools import adfuller
def adf_test(ts, signif=0.05):
    dftest = adfuller(ts, autolag='AIC')
    adf = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags','# Observations'])
    for key,value in dftest[4].items():
       adf['Critical Value (%s)'%key] = value
    print (adf)
    
    p = adf['p-value']
    if p <= signif:
        print(f" Series is Stationary")
    else:
        print(f" Series is Non-Stationary")


# In[ ]:


ns


# In[ ]:


df_differenced


# In[ ]:


X_train.shape


# In[ ]:


for ea_col in X.columns: 
    print(ea_col)
    print("")
    display(adf_test(X[ea_col]))


# In[ ]:


# import statsmodels.tsa.stattools.adfuller
# 1st difference
df_differenced = df_train.diff().dropna()
# stationarity test again with differenced data
display(adf_test(df_differenced["EIA"]))


# In[ ]:


df_diff2 = df_differenced.diff().dropna()
# display(adf_test(df_diff2['146904']))

for ea_col in df_diff2.columns: 
    print(ea_col)
    print("")
    display(adf_test(df_diff2[ea_col]))


# In[ ]:


sns.distplot(df_differenced['EIA'], kde=False, rug=False, fit = stats.norm).set_title('1st Difference')


# In[ ]:


sns.distplot(df_diff2['EIA'], kde = False, rug=False, fit =stats.norm).set_title('2nd Difference')


# In[ ]:


del df_diff2['89088']
del df_diff2['123086']
del df_diff2['408920']
del df_diff2['146847']


# In[ ]:


# del df_differenced['146904']
# del df_differenced['89088']
# del df_differenced['123086']
# del df_differenced['408920']
# del df_differenced['146847']

df_differenced.to_csv('df_differenced.csv')
test = results.test_normality()

# test.HypothesisTestResults()


# In[ ]:


# model fitting
model = VAR(df_differenced)

results = model.fit(ic='aic')
results.summary()


# In[ ]:


plt.title('IQR Plot for all Test Set predictors')
sns.boxplot(x=df_test.values)
plt.show()


# In[ ]:


# forecasting
lag_order = results.k_ar

# lag_order
results.forecast(results.endog, 1)


# In[ ]:


# plotting
results.plot_forecast(20)


# In[ ]:


# Evaluation
fevd = results.fevd(5)
fevd.summary()


# In[ ]:


# forecasting
pred = results.forecast(results.y, steps=nobs)
df_forecast = pd.DataFrame(pred, index=df.index[-nobs:], columns=df.columns + '_1d')
df_forecast.tail()
# inverting transformation
def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_1d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc
# show inverted results in a dataframe
df_results = invert_transformation(df_train, df_forecast, second_diff=True)        
df_results.loc[:, ['realgdp_forecast', 'realcons_forecast']]

