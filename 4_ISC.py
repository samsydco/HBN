#!/usr/bin/env python3

import h5py
import glob
import numpy as np
import random
from settings import *

from brainiak import isfc
ISCf = 'ISC2.h5'

subord = glob.glob(h5path+'sub*.h5')
with h5py.File(h5path+ISCf) as hf:
    string_dt = h5py.special_dtype(vlen=str)
    hf.create_dataset("subord",data=np.array(subord).astype('|S71'),dtype=string_dt)
    
for task in ['DM','TP']:
    data = []
    for sub in subord:
        f = h5py.File(sub, 'r')
        D = {}
        for hem in ['L','R']:
            D[hem] = f[task][hem][:]
        data.append(np.concatenate(
            [D['L'], D['R']], axis=0))
    data = np.dstack(data)
    ISC = isfc.isc(data,collapse_subj=True)
    ISC_persubj = isfc.isc(data,collapse_subj=False)
    with h5py.File(h5path+ISCf) as hf:
        grp = hf.create_group(task)
        grp.create_dataset('ISC', data=ISC)
        grp.create_dataset('ISC_persubj', data=ISC_persubj)