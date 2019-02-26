#!/usr/bin/env python3

import h5py
import os
import glob
import numpy as np
from scipy.stats import zscore
from scipy.spatial.distance import squareform
from settings import *

from brainiak import isfc

ISCf = 'ISC_cat.h5'
if os.path.exists(h5path+ISCf):
    os.remove(h5path+ISCf)

subord = glob.glob(h5path+'sub*.h5')
subord = subord[0:5] # for testing!
ROIs = ['RSC','A1']
TRs = [750,250]
datacat = []
ROIcat = {}
for idx,task in enumerate(['DM','TP']):
    data = []
    ROIcat[task] = {}
    for roi in ROIs:
        ROIcat[task][roi] = []
    for subidx, sub in enumerate(subord):
        f = h5py.File(sub, 'r')
        D = {}
        for hem in ['L','R']:
            D[hem] = f[task][hem][:]
        data.append(np.concatenate([D['L'], D['R']], axis=0))
        for roi in ROIs:
            ROIcat[task][roi].append(zscore(np.mean(f[task][roi][:],axis=0),ddof=1))
    datacat.append(np.dstack(data))
# MIGHT NEED TO Z SCORE BEFORE CONCATENATING...AGAIN??
data = np.concatenate((datacat[0],datacat[1]),axis=1)
ROIiscs = {}
for roi in ROIs:
    ROIiscs[roi] = squareform(np.corrcoef(np.concatenate((np.stack(ROIcat['DM'][roi]),np.stack(ROIcat['TP'][roi])),axis=1)), checks=False)

ISC_persubj = isfc.isc(data,collapse_subj=False)
    
with h5py.File(h5path+ISCf) as hf:
    hf.create_dataset('ISC_persubj', data=ISC_persubj)
    isc_pairs = hf.create_group('isc_pairs')
    for roi in ROIs:
        isc_pairs.create_dataset(roi,data=ROIiscs[roi])
    string_dt = h5py.special_dtype(vlen=str)
    hf.create_dataset("subord",data=np.array(subord).astype('|S71'),dtype=string_dt)


