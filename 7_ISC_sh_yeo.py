#!/usr/bin/env python3

# Add more shuffles for ROIs with low p-values:

import glob
import tqdm
import numpy as np
import deepdish as dd
from scipy.stats import zscore, pearsonr
from ISC_settings import *

bins = [0,4]
nbins = len(bins)
HMMdir = HMMpath+'shuff/'
savedir = ISCpath+'shuff_Yeo/'
nshuff2=1000

def p_calc(ISC):
	nshuff = ISC.shape[0]-1
	p = np.sum(abs(np.nanmean(ISC[0]))<np.nanmean(ISC[1:],axis=1))/nshuff
	return p,nshuff

def load_D(roil,task,bins):
	D = []
	Age = []
	Sex = []
	for bi,b in enumerate(bins):
		bstr = 'bin_'+str(b)
		subl = dd.io.load(roil,'/'+'/'.join([task,bstr,'subl']))
		Sex.extend([Phenodf['Sex'][Phenodf['EID'] == shortsub(sub)].iloc[0] for sub in subl])
		Age.extend([bi]*len(subl))
		D.append(dd.io.load(roil,'/'+'/'.join([task,bstr,'D'])))
	D = np.concatenate(D)
	return D,Age,Sex

def shuff_demo(Age,Sex):
	# Now shuffle Age, and Sex in same order:
	neword = np.random.permutation(len(Age))
	Age = [Age[neword[ai]] for ai,a in enumerate(Age)]
	Sex = [Sex[neword[ai]] for ai,a in enumerate(Sex)]
	return Age,Sex
	
def ISC_w_calc(D,n_vox,n_time,nsub,subh):
	ISC_w = np.zeros((nbins,n_vox))
	groups = np.zeros((nbins,2,n_vox,n_time),dtype='float16')
	for h in [0,1]:
		for htmp in [0,1]:
			group = np.zeros((n_vox,n_time),dtype='float16')
			groupn = np.ones((n_vox,n_time),dtype='int')*nsub//2
			for i in subh[h][htmp]:
				group = np.nansum(np.stack((group,D[i])),axis=0)
				nanverts = np.argwhere(np.isnan(D[i,:]))
				groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
			groups[h,htmp] = zscore(group/groupn,axis=1)
		ISC_w[h] = np.sum(np.multiply(groups[h,0],groups[h,1]), axis=1)/(n_time-1)
	return ISC_w,groups
	
for roi in tqdm.tqdm(glob.glob(savedir+'*.h5')):
	roi_short = roi.split('/')[-1][:-3]
	roil = HMMdir+roi_short+'.h5'
	roidict = dd.io.load(roi)
	for task,taskv in roidict.items():
		e_p,nshuff_e = p_calc(taskv['ISC_e'])
		g_p,nshuff_g = p_calc(taskv['ISC_g'])
		# e_diff test:
		if e_p < 0.05:
			D,Age,Sex = load_D(roil,task,bins)
			nsub_,n_vox,n_time = D.shape
			nsub=nsub_//2
			roidict[task]['ISC_e'] = np.append(taskv['ISC_e'], np.zeros((nshuff2-nshuff_e,n_vox)), axis=0)
			for shuff in np.arange(nshuff_e+1,nshuff2):
				Age,Sex = shuff_demo(Age,Sex)
				subh = even_out(Age,Sex)
				ISC_w,_ = ISC_w_calc(D,n_vox,n_time,nsub,subh)
				roidict[task]['ISC_e'][shuff] = ISC_w[0] - ISC_w[1]
		if g_p < 0.05:
			if len(roidict[task]['ISC_e'])<nshuff2+1:
				D,Age,Sex = load_D(roil,task,bins)
				nsub_,n_vox,n_time = D.shape
				nsub=nsub_//2
			roidict[task]['ISC_g'] = np.append(taskv['ISC_g'], np.zeros((nshuff2-nshuff_g,n_vox)), axis=0)
			for shuff in np.arange(nshuff_g+1,nshuff2):
				Age,Sex = shuff_demo(Age,Sex)
				subh = even_out(Age,Sex)
				ISC_w,groups = ISC_w_calc(D,n_vox,n_time,nsub,subh)
				ISC_b = []
				for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						ISC_b.append(np.sum(np.multiply(groups[0,htmp1], groups[1,htmp2]),axis=1)/(n_time-1))
				roidict[task]['ISC_g'][shuff] = np.sum(ISC_b, axis=0)/4/(np.sqrt(ISC_w[0]) * np.sqrt(ISC_w[1]))	
	dd.io.save(roi,roidict)
				
			
	
