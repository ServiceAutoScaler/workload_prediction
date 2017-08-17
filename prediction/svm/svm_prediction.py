# Importing Libraries
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from statsmodels.tsa.seasonal import seasonal_decompose

from tool.processing import *


# define base model
def svm_model():
    # create model

    return SVR(C=1, cache_size=500, epsilon=1, kernel='rbf')


# Importing dataset
train = pd.read_csv('/Users/tshang/mywork/mypy/table.csv')
timeseries = train['Close'].iloc[2000:5000].iloc[::5].iloc[::-1].values

timeseries=savgol_filter(timeseries, 19, 2)

result = seasonal_decompose(timeseries, model='additive', freq=1)
result.plot().show()

#plt.plot(timeseries)
#plt.show(block=True)
X_WINDOW_SIZE = 9
Y_WINDOW_SIZE = 1
LAG_SIZE = 1
X, Y = split_into_chunks(timeseries, X_WINDOW_SIZE, Y_WINDOW_SIZE, LAG_SIZE, binary=False, scale=True)
X, Y = np.array(X), np.array(Y)

# Splitting the dataset into the Training set and Test set

X#_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
#estimator = KerasRegressor(build_fn=mlp_model, nb_epoch=100, batch_size=5, verbose=0)
estimators = []
#estimators.append(('minmax', MinMaxScaler()))
estimators.append(('standardize', StandardScaler()))
estimators.append(('svr', svm_model()))
pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10, random_state=seed)

results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_absolute_error')

print("Results: %.5f (%.5f) MSE" % (results.mean(), results.std()))



