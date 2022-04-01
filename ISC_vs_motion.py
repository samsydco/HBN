#!/usr/bin/env python3

# ISC vs FD (Motion)
# Compute LOO ISC for each subject in Young group, 
# compare this to their FD
# Are the two values negatively correlated?
# Do this only in parcels that significantly increase in ISC with age


import tqdm
from HMM_settings import *
from motion_check import outliers
import deepdish as dd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

subl = ageeq[0][1][0]+ageeq[1][1][0]
nsub = len(subl)

roidict = dd.io.load(pvals_file,'/roidict')
vals = dd.io.load(pvals_file,'/seeddict/0')

ROIl = []
for roi in roidict.keys():
	if roidict[roi]['ISC_e']['q'] < 0.05 and roidict[roi]['ISC_e']['val'] < 0:
		ROIl.append(roi)

		
FDvsISC = {k:{} for k in ROIl}
for roi in tqdm.tqdm(ROIl):
	hemi = roi[0]
	vall = vals[roi]['vall']
	D = np.empty((nsub,len(vall),750),dtype='float16')
	FD = np.empty((nsub,750),dtype='float16')
	badvox = []
	for sidx, sub in enumerate(subl):
		D[sidx,:,:] = dd.io.load(sub,['/DM/'+hemi],sel=dd.aslice[vall,:])[0]
		FD[sidx] = dd.io.load(sub,['/DM/reg'])[0][:,2]
		badvox.extend(np.where(np.isnan(D[sidx,:,0]))[0]) # Some subjects missing some voxels
	D = np.delete(D,badvox,1)
	vall = np.delete(vall,badvox)
	ISC = np.empty(nsub)
	for sidx in range(nsub):
		LOsub = np.mean(D[sidx],axis=0)
		LIsub = np.mean(np.mean(np.delete(D, sidx, axis=0), axis=0),axis=0)
		ISC[sidx] = np.corrcoef(LOsub,LIsub)[0,1]
	FDvsISC[roi]['FD'] = np.median(FD,axis=1)
	FDvsISC[roi]['ISC'] = ISC
	
dd.io.save(ISCpath+'ISC_vs_motion_outlier.h5',FDvsISC)

p_vals = []	
for roi in ROIl:
	r,p = pearsonr(FDvsISC[roi]['FD'],FDvsISC[roi]['ISC'])
	p_vals.append(p)

q_vals = FDR_p(np.array(p_vals))
for ri,roi in enumerate(ROIl):
	r,p = pearsonr(FDvsISC[roi]['FD'],FDvsISC[roi]['ISC'])
	print(roi+' r = '+str(np.round(r,2))+' p = '+str(np.round(q_vals[ri],2)))
	df=pd.DataFrame({'FD':FDvsISC[roi]['FD'],'ISC':FDvsISC[roi]['ISC']})
	fig, ax = plt.subplots(figsize=(5, 5))	
	ax = sns.regplot(x='FD', y='ISC', data=df)
	ax.set_title(roi+' r = '+str(np.round(r,2))+' p = '+str(np.round(q_vals[ri],2)))
	
	
	
	
	fig, ax = plt.subplots(figsize=(5, 5))	
	ax.plot(FDvsISC[roi]['FD'],FDvsISC[roi]['ISC'],'.')
	ax.set_title(roi+' r = '+str(np.round(r,2)))
	ax.set_xlabel('FD')
	ax.set_ylabel('ISC')
	

	
	
		

	