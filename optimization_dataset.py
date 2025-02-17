import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

normalize_value = 1641775


def load(data):
    non_eq_cl_list = []
    eq_cl_list = []

    data = pd.read_csv("Data/" + data + ".csv")

    temp = (data['Temperature']).tolist()

    for i in range(data.shape[0]):
        non_eq_cl = []
        eq_cl = []

        for kk in range(29):
            non_eq_cl.extend(data.loc[i, ['non_eq_cl'f"{kk}"]])
            eq_cl.extend(data.loc[i, ['eq_cl'f"{kk}"]])

        non_eq_cl[:] = [i / normalize_value for i in non_eq_cl]
        eq_cl[:] = [i / normalize_value for i in eq_cl]

        non_eq_cl_list.append(non_eq_cl)
        eq_cl_list.append(eq_cl)

    non_eq_cl_arr = np.array(non_eq_cl_list)
    eq_cl_ar = np.array(eq_cl_list)

    cl_data = np.append(non_eq_cl_arr, eq_cl_ar, axis=1)

    return cl_data


cl_data = load("opt")

ndim = 4

target = np.array(cl_data)

np.save("target.npy", target)
