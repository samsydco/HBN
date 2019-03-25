#!/usr/bin/env python3

import h5py
import os
import glob
import numpy as np
import deepdish as dd
from scipy.stats import zscore
from scipy.spatial.distance import squareform
from settings import *

from brainiak import isfc
ISCf = 'ISC.h5'
if os.path.exists(ISCpath+ISCf):
    os.remove(ISCpath+ISCf)

subord = glob.glob(prepath+'sub*.h5')
#subord = subord[0:5] # for testing!
ROIs = ['RSC','A1']
for task in ['DM','TP']:
    data = []
    ROIdata = {}
    for roi in ROIs:
        ROIdata[roi] = []
    for subidx, sub in enumerate(subord):
        f = h5py.File(sub, 'r')
        D = {}
        for hem in ['L','R']:
            D[hem] = f[task][hem][:]
        data.append(np.concatenate([D['L'], D['R']], axis=0))
        for roi in ROIs:
            ROIdata[roi].append(zscore(np.mean(f[task][roi][:],axis=0),ddof=1))
    data = np.dstack(data)
    ROIiscs = {}
    for roi in ROIs:
        ROIiscs[roi] = squareform(np.corrcoef(np.stack(ROIdata[roi])), checks=False)
    # ISC = isfc.isc(data,return_p=False,collapse_subj=True)
    ISC_persubj = isfc.isc(data,collapse_subj=False)
    
    with h5py.File(ISCpath+ISCf) as hf:
        grp = hf.create_group(task)
        #grp.create_dataset('ISC', data=ISC)
        grp.create_dataset('ISC_persubj', data=ISC_persubj)
        isc_pairs = grp.create_group('isc_pairs')
        for roi in ROIs:
            isc_pairs.create_dataset(roi,data=ROIiscs[roi])
    
with h5py.File(ISCpath+ISCf) as hf:
    string_dt = h5py.special_dtype(vlen=str)
    hf.create_dataset("subord",data=np.array(subord).astype('|S71'),dtype=string_dt)


    '''
    #pairs = list(itertools.combinations(range(len(subord)), 2))
    voxel_iscs = []
    for v in np.arange(data.shape[0]):
        voxel_data = data[v, :, :].T
        iscs = squareform(np.corrcoef(voxel_data), checks=False)
        # iscs[int(comb(10,2)-comb(10-i,2) + (j-i-1))] 
        voxel_iscs.append(iscs)
    iscs = np.column_stack(voxel_iscs)
    '''
