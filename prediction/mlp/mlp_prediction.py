# Importing Libraries
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD

from tool.processing import *


# define base model
def mlp_model():
    # create model
    model = Sequential()
    model.add(Dense(5, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
    epochs = 50
    learning_rate = 0.3
    decay_rate = learning_rate / epochs
    momentum = 0.2
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=1e-6, nesterov=False)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model


# Importing dataset
train = pd.read_csv('/Users/tshang/mywork/mypy/train_1.csv').fillna(0)
page = train['Page']

# print(train.head(1))
# Dropping Page Column
train = train.drop('Page', axis=1)

# print(page.count)

# Using Data From Random Row for Training and Testing
timeseries = train.iloc[89508, :].values

#plt.plot(timeseries)
#plt.show(block=True)
X_WINDOW_SIZE = 9
Y_WINDOW_SIZE = 1
LAG_SIZE = 1
X, Y = split_into_chunks(timeseries, X_WINDOW_SIZE, Y_WINDOW_SIZE, LAG_SIZE, binary=False, scale=True)
X, Y = np.array(X), np.array(Y)

# Splitting the dataset into the Training set and Test set

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
#estimator = KerasRegressor(build_fn=mlp_model, nb_epoch=100, batch_size=5, verbose=0)
estimators = []
#estimators.append(('minmax', MinMaxScaler()))
#estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=mlp_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10, random_state=seed)

results = cross_val_score(pipeline, X, Y, cv=kfold,scoring='neg_mean_absolute_error')

print("Results: %.5f (%.5f) MSE" % (results.mean(), results.std()))


# model.fit(X_train,
#           Y_train,
#           epochs=50,
#           batch_size=5,
#           verbose=0,
#           validation_split=0.1)
# score = model.evaluate(X_test, Y_test, batch_size=5, show_accuracy=True, verbose=0)
# print(score)
