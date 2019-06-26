#!/usr/bin/env python3

import h5py
import deepdish as dd
import pandas as pd
import os
import glob
import tqdm
import numpy as np
from tqdm import tqdm
from scipy import stats
from random import shuffle
from scipy.spatial.distance import squareform
from settings import *
from brainiak import isfc

ROIf = 'ROIstack'
if os.path.exists(ISCpath+ROIf+'.h5'):
	os.remove(ISCpath+ROIf+'.h5')

# simple, one roi isc function
def isc(D):
	n_subj = D.shape[0]
	ISC = np.zeros(n_subj)
	for loo_subj in range(n_subj):
		group = np.mean(D[np.arange(n_subj) != loo_subj, :], axis=0)
		subj = D[loo_subj,:]
		ISC[loo_subj] = stats.pearsonr(group,subj)[0]
	return ISC

def isc_space(D):
	n_subj,n_vox,n_time = D.shape
	ISC = np.zeros((n_subj,n_time))
	for loo_subj in range(n_subj):
		group = stats.zscore(np.mean(D[np.arange(n_subj) != loo_subj, :,:], axis=0),axis=0)
		subj = stats.zscore(D[loo_subj,:,:],axis=0)
		ISC[loo_subj,:] = np.sum(np.multiply(group,subj),axis=0)/(n_vox-1)

# create list of per-demo, per-ROI iscs, and compare with parametric t-test
roiiscdict = {}
for f in tqdm(glob.glob(ISCpath+ROIf+'_*')):
	roi = f.split('/')[-1].split(ROIf+'_')[-1].split('.h5')[0]
	ROIdata = dd.io.load(f,['/'+roi[3:]])[0]
	if np.sum(np.isnan(ROIdata))>0 and np.sum(np.isnan(ROIdata[:,0,0]))<len(subord):
		# sub: sub-NDARAX283MAK, TP, and sub-NDARYY218LU2, TP, missing some values in files
		# TP_ISC_L.3 and TP_ISC_R.2
		ROIdata[np.where(np.isnan(ROIdata))] = 0
	ROIdata = stats.zscore(ROIdata,axis=2,ddof=1)
	roiiscdict[roi] = {}
	roiiscdict[roi]['xcorr'] = squareform(np.corrcoef(np.mean(ROIdata,axis=1)), checks=False)
	for p, v in phenol.items():
		roiiscdict[roi][p] = {}
		roiiscdict[roi][p]['isc_true'] = isc(np.mean(ROIdata,axis=1)[[i == True for i in v],:])
		roiiscdict[roi][p]['isc_false'] = isc(np.mean(ROIdata,axis=1)[[i == False for i in v],:])
		roiiscdict[roi][p]['stats'] = stats.ttest_ind(roiiscdict[roi][p]['isc_true'],roiiscdict[roi][p]['isc_false'])
		roiiscdict[roi][p]['patt_true'] = isc_space(ROIdata[[i == True for i in v],:,:])
		roiiscdict[roi][p]['patt_false'] = isc_space(ROIdata[[i == False for i in v],:,:])
		#print(p,'\n',roi,'\n',roiiscdict[roi][p]['stats'])
'''
for roi,vals in roiiscdict.items():
	for dem in list(phenol.keys()):
		if vals[dem][-1][-1] < 0.05:
			mean1 = round(np.mean(vals[dem]['isc_true']),3)
			mean2 = round(np.mean(vals[dem]['isc_false']),3)
			p = vals[dem]['stats'][-1]
			print(roi,'\n',dem,'\n','mean1 = ',mean1,'mean2 = ',mean2,'\n',p)
'''

with h5py.File(ISCpath+ROIf+'.h5') as hf:
	for p, v in roiiscdict.items():
		grp = hf.create_group(p)
		for i,vv in enumerate(v):
			grp.create_dataset('isc'+'_'+str(i),data=vv)

# Actual p-values and means in poster submission:
# bilateral TP ROIs permutation test
SUMAloc = ISCpath+ROIf+'_DM_ISC_SUMA_'
SUMAROIs = [f.strip(SUMArois)[1:-1] for f in [f for f in glob.glob(SUMAloc+'*') if '_R' not in f]]
for roi in SUMAROIs:
	f = []
	for task in ['TP']:
		SUMAloc = ISCpath+ROIf+'_'+task+'_ISC_SUMA_'
		f.append(np.concatenate((dd.io.load(SUMAloc+'L'+roi+'.h5',['/ISC_SUMA_L'+roi])[0],
							dd.io.load(SUMAloc+'R'+roi+'.h5',['/ISC_SUMA_R'+roi])[0]),
						   axis=1))
	f = np.mean(stats.zscore(f[0],axis=2,ddof=1),axis=1)
	#f = np.mean(stats.zscore(np.concatenate((f[0],f[1]),axis=2),axis=2,ddof=1),axis=1)=
	ids = np.arange(len(phenol['age']))
	diff = []
	for i in range(10001):
		nl = [x for _,x in sorted(zip(ids,phenol['age']))]
		diff.append(np.mean(isc(f[[i == True for i in nl],:]))-np.mean(isc(f[[i == False for i in nl],:])))
		shuffle(ids)
	p = np.mean(abs(diff[0]) < [abs(i) for i in diff[1:]])
	print(roi,'\n','age','\n','mean1 = ',round(np.mean(isc(f[[i == True for i in phenol['age']],:])),3),'mean2 = ',round(np.mean(isc(f[[i == False for i in phenol['age']],:])),3),'\n',p)
			
			

		
		

			

