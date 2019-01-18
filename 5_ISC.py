#!/usr/bin/env python3

import h5py
import glob
import numpy as np
from scipy.spatial.distance import squareform
from settings import *

from brainiak import isfc
ISCf = 'ISC.h5'

subord = glob.glob(h5path+'sub*.h5')
#subord = subord[0:5]
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
            ROIdata[roi].append(f[task][roi][:])
    data = np.dstack(data)
    ROIiscs[roi]
    for roi in ROIs:
        ROIiscs[roi] = squareform(np.corrcoef(np.stack(ROIdata[roi]), checks=False))
    ISC = isfc.isc(data,return_p=False,collapse_subj=True)
    ISC_persubj = isfc.isc(data,collapse_subj=False)
    with h5py.File(h5path+ISCf) as hf:
        grp = hf.create_group(task)
        grp.create_dataset('ISC', data=ISC)
        grp.create_dataset('ISC_persubj', data=ISC_persubj)
        if 'isc_pairs' in list(f[task].keys()):
                    del f[task]['isc_pairs']
        isc_pairs = grp.create_group('isc_pairs',data=iscs)
        for roi in ROIs:
            isc_pairs.create_dataset(roi,data=ROIiscs[roi])
    
with h5py.File(h5path+ISCf) as hf:
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
