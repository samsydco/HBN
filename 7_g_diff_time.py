#!/usr/bin/env python3

# Look at g_difference over time in ROIs with big g_difference:
# at every time point calculate:
# Z-scored(young)*Z-scored(old)
import os
import tqdm
import glob
import numpy as np
import deepdish as dd
from datetime import date
from scipy.stats import zscore
import matplotlib.pyplot as plt
from settings import *
from ISC_settings import *

gdiffroidir = path+'ROIs/SfN_2019/Fig3_'#'ROIs/g_diff/'
g_diff_f = ISCpath+'g_diff_time_'+str(date.today())+'.h5'
if os.path.exists(g_diff_f):
    os.remove(g_diff_f)
nTR=750
ROIl = glob.glob(gdiffroidir+'*roi')
nROI = len(ROIl)
gdict = {}
for f in tqdm.tqdm(ROIl):
	fn = f.split(gdiffroidir)[1]
	roin = fn[:-7]
	task = 'DM' if fn[:2] == 'TP' else 'TP'
	hemi = fn[3]
	roi = np.loadtxt(f).astype('int')
	vall = roi[roi[:,1]==1,0]
	n_vox = len(vall)
	groups = np.zeros((2,n_vox,nTR),dtype='float16')
	for i,b in enumerate([0,nbinseq-1]):
		subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
		# Load data and z-score within a group:	
		group = np.zeros((n_vox,nTR),dtype='float16')
		groupn = np.ones((n_vox,nTR),dtype='int')*len(subl)
		for sub in subl:
			d = dd.io.load(sub,['/'+task+'/'+hemi],sel=dd.aslice[vall,:])[0]
			group = np.nansum(np.stack((group,d)),axis=0)
			nanverts = np.argwhere(np.isnan(d))
			groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
		groups[i] = zscore(group/groupn,axis=0)
	gdict[roin] = np.sum(np.multiply(groups[0],groups[1]),axis=0)/(n_vox-1)
dd.io.save(g_diff_f,gdict)
xticks = np.arange(0,600,0.8) #x axis in seconds
for s in [0,10,20,30]:
	plt.figure()
	ham = np.hamming(np.round(s/0.8))
	ham = ham/np.sum(ham)
	plt.style.use('seaborn-muted')
	for r,t in gdict.items():
		if ham.size==0:
			plt.plot(xticks,t,label=r[6:])
		else:
			plt.plot(xticks,np.convolve(t,ham,'same'),label=r[6:])
		plt.xlabel('time (sec)')
		plt.ylabel('ISC (r)')
	plt.legend()
	plt.tight_layout()
	plt.show
	plt.savefig(figurepath+'g_diff_time/DM_blur'+str(s)+'.png')

	
	
	
		
		
	

