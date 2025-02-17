import pandas as pd
import numpy as np
import os
import tensorflow as tf


def load(dataset):
    Npivot = 30
    temp = []
    features = []
    data = pd.read_csv("Data/weighted_data/" + dataset + ".csv")

    temp = (data['Temperature']).tolist()

    intervals = (data['Intervales']).tolist()

    for i in range(data.shape[0]):
        feature = []
        m1_feed_rate = []
        m2_feed_rate = []

        for ii in range(intervals[i], intervals[i] - 4, -1):
            data.loc[i, ['M1fed'f"{ii}"]] = -1
            data.loc[i, ['M2fed'f"{ii}"]] = -1

        for jj in range(intervals[i]):
            m1_feed_rate.extend(data.loc[i, ['M1fed'f"{jj}"]])
            m2_feed_rate.extend(data.loc[i, ['M2fed'f"{jj}"]])

            feature.append([temp[i], data.at[i, 'M1fed'f"{jj}"], data.at[i, 'M2fed'f"{jj}"]])
        features.append(feature)

    padded_data = tf.keras.preprocessing.sequence.pad_sequences(features, padding='post', dtype=float)
    return padded_data


ndim = 4

data = load("test")

np.save("data.npy", data)
