# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:07:47 2020

@author: Jérôme
"""

import pandas as pd
import numpy as np
from make_graphs import ecg_plot, eeg_plot
import scipy.ndimage as im
from sklearn.decomposition import FastICA

# -------------------------------------------------------------------------------------\
# 1. Read the data
# -------------------------------------------------------------------------------------\
    
nb_channels = 32
ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2']
data = pd.read_table('Data/EEG_data.csv', sep=",", header=None)
eeg = np.transpose(data.values)

#print(type(eeg))
#O1 = np.ndarray(shape = (1, eeg.shape[1]))
#O1[0] = eeg[-1]
#ecg_plot(eeg, True, ch_names)

filtered_data = np.ndarray(eeg.shape)

#filtered_data = np.ndarray(shape=(1, eeg.shape[1]))
#filtered_data[0] = im.gaussian_filter(O1[0], 200)
#
for i in range(eeg.shape[0]):
    filtered_data[i] = im.gaussian_filter(eeg[i], 100)
#    
#ecg_plot(filtered_data, True, ch_names)

ica = FastICA(32)
print(ica.fit_transform(filtered_data))