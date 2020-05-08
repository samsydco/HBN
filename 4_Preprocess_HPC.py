#!/usr/bin/env python3

import glob
import nibabel as nib
import numpy as np
import os
import h5py
from sklearn import linear_model
from scipy import stats, special
from settings import *

subs = glob.glob('%ssub*.html'%(fmripreppath))
subs = [s.replace('.html', '').replace(fmripreppath, '') for s in subs]
subs = [sub for sub in subs if not os.path.isfile(hpcprepath + sub + '.h5') and sub not in bad_sub_dict]

dpath = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
mask_path = '_space-MNI152NLin2009cAsym_desc-aseg_dseg.nii.gz'
conf_path = '_desc-confounds_regressors.tsv'

for sub in subs:
	print('Processing subject ', sub)
	for task in ['DM','TP']:
		print('movie ', task)
		fpath = fmripreppath+sub+'/func/'+sub+'_task-movie'+task
		nii = nib.load(fpath+dpath).get_data()
		mask = nib.load(fpath+mask_path).get_data()
		mask = np.logical_or(mask == 17, mask == 53)
		hipp = nii[mask]

		# Use regressors for:
		# -CSF
		# -WhiteMatter
		# -FramewiseDisplacement
		# -All cosine bases for drift (0.008 Hz = 125s)
		# -X, Y, Z and derivatives
		# -RotX, RotY, RotZ and derivatives
		
		conf = np.genfromtxt(fpath+conf_path, names=True)
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
		with h5py.File(os.path.join(hpcprepath + sub + '.h5')) as hf:
			grp = hf.create_group(task)
			grp.create_dataset('HPC', data=hipp)
			grp.create_dataset('reg',data=reg)