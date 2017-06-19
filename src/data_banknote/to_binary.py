#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import csv
import numpy as np


# load data from txt/csv
data = np.loadtxt('data_banknote_authentication.txt',delimiter=',')

print(data.shape)
rows = data.shape[0]   # number of data points
cols = data.shape[1]-1 # number of features in each datapoint (last column of data describes classes)

with open('data_banknote_authentication.info','wb') as ofile:
    writer = csv.writer(ofile,delimiter=' ',quotechar=' ')
    writer.writerow(['rows:',rows])
    writer.writerow(['cols:',cols])
    ofile.close()

# save data to file
data[:,:4].T.astype('float32').tofile('data_banknote_authentication_flt.bin')
data[:,:4].T.astype('float64').tofile('data_banknote_authentication_dbl.bin')

classes = data[:,4]
classes.astype('uint8').tofile('classes.bin')