#!/usr/bin/env python3

import h5py
import glob
import numpy as np
import os
import pandas as pd
import math
from scipy.spatial.distance import squareform
from settings import *

from brainiak import isfc
ISCf = 'ISCtest.h5'

subord = glob.glob(h5path+'sub*.h5')
#subord = subord[0:5]

# If Wronglencsv exists, import info, if not, create it
if (os.path.exists(h5path+ISCf) and "/WronglenDF" in h5path+ISCf):
    with h5py.File(h5path+ISCf) as hf:
        df = hf["WronglenDF"][:]
else:
    df = pd.DataFrame(columns=["Subject","DM","TP"])
# Remove sub from dataset if wrong number of TRs
# 
for idx, task in enumerate(['DM','TP']):
    data = []
    badsubs = []
    for subidx, sub in enumerate(subord):
        sub_short = sub[52:]
        if (df.empty or (df['Subject'].isin([sub_short]).any()!=True or ((df.iat[df.loc[df['Subject']==sub_short].index[0],idx+1]==exp_len[idx] or math.isnan(df.iat[df.loc[df['Subject']==sub_short].index[0],idx+1]))))):
            f = h5py.File(sub, 'r')
            D = {}
            for hem in ['L','R']:
                D[hem] = f[task][hem][:]
            NTRs = D[hem].shape[1]
            if df['Subject'].isin([sub_short]).any()==False:
                df = pd.concat([df, pd.DataFrame({'Subject': sub_short,task: NTRs}, index=[0])], ignore_index=True,sort=False)
            else:
                df.iloc[df.loc[df['Subject']==sub_short].index[0], df.columns.get_loc(task)] = NTRs
            if NTRs == exp_len[idx]:
                data.append(np.concatenate([D['L'], D['R']], axis=0))
            else:
                badsubs.append(subidx)
    data = np.dstack(data)
    subord = [i for j, i in enumerate(subord) if j not in badsubs]
    #pairs = list(itertools.combinations(range(len(subord)), 2))
    voxel_iscs = []
    for v in np.arange(data.shape[0]):
        voxel_data = data[v, :, :].T
        iscs = squareform(np.corrcoef(voxel_data), checks=False)
        # iscs[int(comb(10,2)-comb(10-i,2) + (j-i-1))] 
        voxel_iscs.append(iscs)
    iscs = np.column_stack(voxel_iscs)
    ISC = isfc.isc(data,return_p=False,collapse_subj=True)
    ISC_persubj = isfc.isc(data,collapse_subj=False)
    with h5py.File(h5path+ISCf) as hf:
        grp = hf.create_group(task)
        grp.create_dataset('ISC', data=ISC)
        grp.create_dataset('ISC_persubj', data=ISC_persubj)
        grp.create_dataset('isc_pairs',data=iscs)
    
    df[task] = df[task].astype(int)
df.to_hdf(h5path+ISCf, "WronglenDF", table=True, mode='a')
with h5py.File(h5path+ISCf) as hf:
    string_dt = h5py.special_dtype(vlen=str)
    hf.create_dataset("subord",data=np.array(subord).astype('|S71'),dtype=string_dt)

'''
        
'''

