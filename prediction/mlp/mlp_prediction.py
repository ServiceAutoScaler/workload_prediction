# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from .processing import *

# Importing dataset
train = pd.read_csv('/Users/tshang/mywork/mypy/train_1.csv').fillna(0)
page = train['Page']

# print(train.head(1))
# Dropping Page Column
train = train.drop('Page', axis=1)

#print(page.count)

# Using Data From Random Row for Training and Testing
timeseries = train.iloc[67409, :].values

#plt.plot(row)
#plt.show(block=True)
X_WINDOW_SIZE=10
Y_WINDOW_SIZE=1
LAG_SIZE=1
X, Y = split_into_chunks(timeseries, X_WINDOW_SIZE, Y_WINDOW_SIZE, LAG_SIZE, binary=False, scale=True)
X, Y = np.array(X), np.array(Y)


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
