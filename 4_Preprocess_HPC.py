#!/usr/bin/env python3

import glob
import nibabel as nib
import pandas as pd
import numpy as np
import os
import h5py
from sklearn import linear_model
from scipy import stats, special
from settings import *

dpath = '../fmriprep/fmriprep/sub-01/ses-0%d/func/%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
mask_path = '../fmriprep/fmriprep/sub-01/ses-0%d/func/%s_space-MNI152NLin2009cAsym_desc-aseg_dseg.nii.gz'
conf_path = '../fmriprep/fmriprep/sub-01/ses-0%d/func/%s_desc-confounds_regressors.tsv'

for session, run in zip(sessions, run_names):
    nii = nib.load(dpath % (session, run)).get_data()
    mask = nib.load(mask_path % (session, run)).get_data()
    mask = np.logical_or(mask == 17, mask == 53)

    hipp = nii[mask]

    # Use regressors for:
    # -CSF
    # -WhiteMatter
    # -FramewiseDisplacement
    # -All cosine bases for drift (0.008 Hz = 125s)
    # -X, Y, Z and derivatives
    # -RotX, RotY, RotZ and derivatives

    conf = np.genfromtxt(conf_path % (session, run), names=True)
    motion = np.column_stack((conf['trans_x'],
                              conf['trans_y'],
                              conf['trans_z'],
                              conf['rot_x'],
                              conf['rot_y'],
                              conf['rot_z']))

    reg = np.column_stack((conf['csf'],
                           conf['white_matter'],
                           np.nan_to_num(conf['framewise_displacement']),
                           np.column_stack([conf[k] for k in conf.dtype.names if
                                            'cosine' in k]),
                           motion,
                           np.vstack((np.zeros((1, motion.shape[1])),
                                      np.diff(motion, axis=0)))))

    print('      Cleaning and zscoring')
    regr = linear_model.LinearRegression()
    regr.fit(reg, hipp.T)
    hipp = hipp - np.dot(regr.coef_, reg.T) - regr.intercept_[:, np.newaxis]

    hipp = stats.zscore(hipp, axis=1)

    with h5py.File(os.path.join('PythonData', run + '_hipp.h5'), 'w') as hf:
        hf.create_dataset('hipp', data=hipp)