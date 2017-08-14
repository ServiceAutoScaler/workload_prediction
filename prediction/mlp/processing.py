# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing

def split_into_chunks(data, train, predict, step, binary=True, scale=True):
    X, Y = [], []
    for i in range(0, len(data), step):
        try:
            x_i = data[i:i + train]
            y_i = data[i + train + predict]

            # Use it only for daily return time series
            if binary:
                if y_i > 0.:
                    y_i = [1., 0.]
                else:
                    y_i = [0., 1.]

                if scale: x_i = preprocessing.scale(x_i)

            else:
                timeseries = np.array(data[i:i + train + predict])
                if scale: timeseries = preprocessing.scale(timeseries)
                x_i = timeseries[:-1]
                y_i = timeseries[-1]

        except:
            break

        X.append(x_i)
        Y.append(y_i)

    return X, Y


def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def create_train_test(X, y, percentage=0.8):
    X_train = X[0:len(X) * percentage]
    Y_train = y[0:len(y) * percentage]

    X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_test = X[len(X) * percentage:]
    Y_test = y[len(X) * percentage:]

    return X_train, X_test, Y_train, Y_test
