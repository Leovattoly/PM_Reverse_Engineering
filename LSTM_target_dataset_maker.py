import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

NORMALIZE_VALUE = 1641775

def Map_error(user_data, pred_data):
    mae = mean_absolute_error(user_data, pred_data)
    return mae


def rmse(s_data, d_data):
    sd = 0
    for i, j in zip(s_data, d_data):
        sd = sd + (i - j) ** 2 / len(s_data)
    return np.sqrt(sd)


def encoder(ndim):
    encoder_model = pickle.load(open("encoder_model_16500.sav", 'rb'))
    return encoder_model


def load(data):
    non_eq_cl_list = []
    eq_cl_list = []
    data = pd.read_csv("Data/" + data + ".csv")
    print("Shape:",data.shape)

    for i in range(data.shape[0]):
        non_eq_cl = []
        eq_cl = []
        for kk in range(29):
            non_eq_cl.extend(data.loc[i, ['non_eq_cl'f"{kk}"]])
            eq_cl.extend(data.loc[i, ['eq_cl'f"{kk}"]])

        non_eq_cl[:] = [i / NORMALIZE_VALUE for i in non_eq_cl]
        eq_cl[:] = [i / NORMALIZE_VALUE for i in eq_cl]

        non_eq_cl_list.append(non_eq_cl)
        eq_cl_list.append(eq_cl)

    non_eq_cl_arr = np.array(non_eq_cl_list)
    eq_cl_ar = np.array(eq_cl_list)
    cl_arr = np.concatenate((non_eq_cl_arr, eq_cl_ar), axis=1)
    return cl_arr


data = load("test")

ndim = 4

print(data.shape)


# For regression purpose
encoder_model = encoder(ndim)

encoded_data = encoder_model.predict(data)

# For regression purpose
data = np.append(encoded_data, data, axis=1) # followed by the encoded data appended  original data
np.save("target.npy",data)
