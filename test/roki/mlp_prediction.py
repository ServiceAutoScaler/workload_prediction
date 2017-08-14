# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Importing dataset
train = pd.read_csv('/Users/tshang/mywork/mypy/train_1.csv').fillna(0)
page = train['Page']

# print(train.head(1))
# Dropping Page Column
train = train.drop('Page', axis=1)

print(page.count)

# Using Data From Random Row for Training and Testing
row = train.iloc[91000, :].values
X = row[0:549]
y = row[1:550]

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
