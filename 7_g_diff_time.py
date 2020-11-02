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

ROIdir = HMMpath+'shuff_5bins_train04/'
ROIl = [roi.split('/')[-1][:-3] for roi in glob.glob(ROIdir+'*')]
task= 'DM'
nTR=750
bins = np.arange(nbinseq)

def calc_isc(g1,g2,n_vox):
	ISCt = np.sum(np.multiply(g1,g2),axis=0)/(n_vox-1)
	return ISCt

#g_diff_f = ISCpath+'g_diff_time_2019-10-04.h5'
g_diff_f = ISCpath+'g_diff_time_.h5'
gdict = {}
for roi in tqdm.tqdm(ROIl):
	roidict = dd.io.load(ROIdir+roi+'.h5','/'+task)
	D = [np.mean(roidict['bin_'+str(b)]['D'],axis=0).T for b in bins]
	n_vox = len(roidict['vall'])
	groups = np.zeros((2,2,n_vox,nTR),dtype='float16')
	for i,b in enumerate([0,nbinseq-1]):
		subl = [[],[]]
		for ii in [0,1]:
			subg = [ageeq[ii][1][b][idx] for idx in np.random.choice(lenageeq[ii][b],divmod(minageeq[ii],2)[0]*2,replace=False)]
			subl[0].extend(subg[:divmod(minageeq[i],2)[0]])
			subl[1].extend(subg[divmod(minageeq[i],2)[0]:])
		for h in [0,1]:
			# Load data and z-score within a group:	
			group = np.zeros((n_vox,nTR),dtype='float16')
			groupn = np.ones((n_vox,nTR),dtype='int')*len(subl)
			for sub in subl[h]:
				d = dd.io.load(sub,['/'+task+'/'+hemi],sel=dd.aslice[vall,:])[0]
				group = np.nansum(np.stack((group,d)),axis=0)
				nanverts = np.argwhere(np.isnan(d))
				groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
			groups[i,h] = zscore(group/groupn,axis=0)
	ISCg = np.zeros(nTR) 
	for htmp1 in [0,1]:
		for htmp2 in [0,1]:
			ISCg += calc_isc(groups[0,htmp1],groups[1,htmp2],n_vox)
	ISC0 = calc_isc(groups[0,0],groups[0,1],n_vox)
	ISC1 = calc_isc(groups[1,0],groups[1,1],n_vox)
	ISCg[ISCg<0] = 0
	ISC0[ISC0<0] = 0
	ISC1[ISC1<0] = 0
	gdict[roin] = {}
	for s in ses:
		ham = np.hamming(np.round(s/0.8))
		ham = ham/np.sum(ham)
		if s==0:
			gdict[roin][str(s)] = ISCg/4/(np.sqrt(ISC0)*np.sqrt(ISC1))
			gdict[roin][str(s)][~np.isfinite(gdict[roin][str(s)])] = 0
		else:
			gdict[roin][str(s)] = np.convolve(ISCg,ham,'same')/4/(np.sqrt(np.convolve(ISC0, ham, 'same')) * np.sqrt(np.convolve(ISC1, ham, 'same')))
dd.io.save(g_diff_f,gdict)

gdict = dd.io.load(g_diff_f)
xticks = np.arange(0,600,0.8) #x axis in seconds
for s in ses:
	plt.figure()
	plt.style.use('seaborn-muted')
	for r,t in gdict.items():
		if ham.size==0:
			plt.plot(xticks,t[str(s)],label=r[6:])
		else:
			plt.plot(xticks,np.convolve(t[str(s)],ham,'same'),label=r[6:])
		plt.xlabel('time (sec)')
		plt.ylabel('ISC (r)')
	plt.legend()
	plt.tight_layout()
	plt.show
	plt.savefig(figurepath+'g_diff_time/DM_blur'+str(s)+'.png')

	
	
	
		
		
	

