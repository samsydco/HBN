#!/usr/bin/env python3

import tqdm
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
subs = [sub for sub in subs if sub not in bad_sub_dict]
subs = [sub for sub in subs if not os.path.isfile(hpcprepath + sub + '.h5') and sub not in bad_sub_dict]

mask_path = '_space-MNI152NLin2009cAsym_desc-aseg_dseg.nii.gz'
# Version 1.1.4:
dpath = '_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
conf_path = '_bold_confounds.tsv'

psplit = 45 # voxels less then 45 are in post HPC (-21mm and less in MNI)
asplit = 46 # voxels greater then 45 are in anterior HPC (-20mm and greater in MNI)
# a/p split based on: http://www.dpmlab.org/papers/ENEURO.0178-16.2016.full.pdf
# This is helpful: http://blog.chrisgorgolewski.org/2014/12/how-to-convert-between-voxel-and-mm.html

for sub in tqdm.tqdm(subs):
	for task in ['DM']:
		fpath = fmripreppath+sub+'/func/'+sub+'_task-movie'+task
		nii = nib.load(fpath+dpath).get_data()
		mask = nib.load(outputdrHPC+sub+'/func/'+sub+'_task-movie'+task+mask_path)
		mask = mask.get_data()
		mask = np.logical_or(mask == 17, mask == 53)
		apmask = np.concatenate([np.full(mask[:,:psplit,:].shape,1), np.zeros((mask.shape[0],1,mask.shape[0]),dtype=bool), np.full(mask[:,asplit:,:].shape,2)],axis=1)
		hipp = nii[mask]
		aplab = apmask[mask] #1=posterior, #2=anterior

		# Use regressors for:
		# -CSF
		# -WhiteMatter
		# -FramewiseDisplacement
		# -All cosine bases for drift (0.008 Hz = 125s)
		# -X, Y, Z and derivatives
		# -RotX, RotY, RotZ and derivatives
		
		conf = np.genfromtxt(fpath+conf_path, names=True)
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
                           np.vstack((np.zeros((1,motion.shape[1])), 
									  np.diff(motion, axis=0)))))

		regr = linear_model.LinearRegression()
		regr.fit(reg, hipp.T)
		hipp = hipp - np.dot(regr.coef_, reg.T) - regr.intercept_[:, np.newaxis]
		hipp = stats.zscore(hipp, axis=1)
		with h5py.File(os.path.join(hpcprepath + sub + '.h5')) as hf:
			grp = hf.create_group(task)
			grp.create_dataset('HPC', data=hipp)
			grp.create_dataset('aplab', data=aplab)
			grp.create_dataset('reg',data=reg)