#!/usr/bin/env python3

import glob
import nibabel as nib
import pandas as pd
import numpy as np
import os
import h5py
from sklearn import linear_model
from scipy import stats
from settings import *

subs = glob.glob('%ssub*.html'%(fmripreppath_old))
subs = [s.replace('.html', '') for s in subs]
subs = [s.replace(fmripreppath_old, '') for s in subs]
subs = [sub for sub in subs if not os.path.isfile(prepath + sub + '.h5') and sub not in bad_sub_dict]
# Check if fmap has been processed
# subs = [sub for sub in subs if len(glob.glob(fmripreppath_old+sub+'/figures/*sdc*'))==2]

Phenodf = pd.concat((pd.read_csv(f) for f in glob.glob(phenopath+'HBN_R*Pheno.csv')),ignore_index=True)

for sub in subs:
    print('Processing subject ', sub)
    Demo = {'Age': Phenodf['Age'][Phenodf['EID'] == sub[4:]].iloc[0],
           'Sex': Phenodf['Sex'][Phenodf['EID'] == sub[4:]].iloc[0]}
    with h5py.File(os.path.join(prepath + sub + '.h5')) as hf:
        grp = hf.create_group('Pheno')
        for k,v in Demo.items():
            grp.create_dataset(k,data=v)
    for task in ['DM','TP']:
        D = dict()
        print('movie ', task)
        for hem in ['L', 'R']:
            fname = os.path.join(fmripreppath_old + sub + '/func/' + \
          sub + '_task-movie' + task + '_bold_space-fsaverage6.' + hem + '.func.gii')
            print('      Loading ', fname)
            gi = nib.load(fname)
            D[hem] = np.column_stack([gi.darrays[t].data for t in range(len(gi.darrays))])
        
        # Use regressors for:
        # -CSF
        # -WhiteMatter
        # -FramewiseDisplacement
        # -All cosine bases for drift (0.008 Hz = 125s)
        # -X, Y, Z and derivatives
        # -RotX, RotY, RotZ and derivatives
    
        conf = np.genfromtxt(os.path.join(fmripreppath_old + sub + '/func/' + \
                  sub + '_task-movie' + task + '_bold_confounds.tsv'), names=True)
        motion = np.column_stack((conf['X'],
                              conf['Y'],
                              conf['Z'],
                              conf['RotX'],
                              conf['RotY'],
                              conf['RotZ']))

        reg = np.column_stack((conf['CSF'],
                           conf['WhiteMatter'],
                           np.nan_to_num(conf['FramewiseDisplacement']),
                           np.column_stack([conf[k] for k in conf.dtype.names if 'Cosine' in k]),
                           motion,
                           np.vstack((np.zeros((1,motion.shape[1])), np.diff(motion, axis=0)))))
                           
        print('      Cleaning and zscoring')
        for hem in ['L', 'R']:
            regr = linear_model.LinearRegression()
            regr.fit(reg, D[hem].T)
            D[hem] = D[hem] - np.dot(regr.coef_, reg.T) - regr.intercept_[:, np.newaxis]
            # Note 8% of values on cortical surface are NaNs, and the following will therefore throw an error
            D[hem] = stats.zscore(D[hem], axis=1)
        with h5py.File(os.path.join(prepath + sub + '.h5')) as hf:
            grp = hf.create_group(task)
            grp.create_dataset('L', data=D['L'])
            grp.create_dataset('R', data=D['R'])
            grp.create_dataset('reg',data=reg)


    