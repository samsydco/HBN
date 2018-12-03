#!/usr/bin/env python3

import glob
import nibabel as nib
import numpy as np
import os
import h5py
from sklearn import linear_model
from scipy import stats,special

path = '/data/HBN/test/fmriprep_output/fmriprep/'
subs = glob.glob('%ssub*.html'%(path))
subs = [s.replace('.html', '') for s in subs]
subs = [s.replace(path, '') for s in subs]

for sub in subs:
  D = dict()
  print('Processing subject ', sub)
  for task in ['DM','TP']:
    print('movie ', task)
    for hem in ['L', 'R']:
        fname = os.path.join(path + sub + '/func/' + \
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
    
    conf = np.genfromtxt(os.path.join(path + sub + '/func/' + \
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

        D[hem] = stats.zscore(D[hem], axis=1)
    
    with h5py.File(os.path.join(path + 'PythonData/' + sub + task + '.h5'), 'w') as hf:
      hf.create_dataset('L', data=D['L'])
      hf.create_dataset('R', data=D['R'])