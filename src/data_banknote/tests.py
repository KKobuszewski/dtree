#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import csv
import numpy as np


# load data from txt/csv
data = np.loadtxt('data_banknote_authentication.txt',delimiter=',')
rows = data.shape[0]   # number of data points
cols = data.shape[1]-1 # number of features in each datapoint (last column of data describes classes)


classes = data[:,-1]
data    = data[:,:-1]

print()
print(classes)
print(data)
print()

x = data[:,0]
max_x = np.max(x)
min_x = np.min(x)
N_SPLITS = 32
dx = (max_x - min_x)/float(N_SPLITS)

print('sum:',np.sum(classes),' fraction:',np.sum(classes)/float(classes.size))
for it in range(N_SPLITS):
    s = np.sum(classes[np.where(x>min_x)])
    fraction = s/float(classes.size)
    print( '{}.\t{}\t'.format(it,s), fraction )
    min_x += dx