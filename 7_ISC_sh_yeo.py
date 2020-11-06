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
	
for roi in tqdm.tqdm(glob.glob(savedir+'*.h5')):
	roi_short = roi.split('/')[-1][:-3]
	roil = HMMdir+roi_short+'.h5'
	roidict = dd.io.load(roi)
	for task in list(roidict.keys()):
		print(roi,task)
		taskv = roidict[task]
		e_p,nshuff_e = p_calc(taskv['ISC_e'],'e')
		g_p,nshuff_g = p_calc(taskv['ISC_g'],'g')
		# e_diff test:
		if e_p < 0.05 and nshuff_e<nshuff2:
			print('e_diff')
			D,Age,Sex = load_D(roil,task,bins)
			nsub_,n_vox,n_time = D.shape
			nsub=nsub_//2
			roidict[task]['ISC_e'] = np.append(taskv['ISC_e'], np.zeros((nshuff2-nshuff_e,n_vox)), axis=0)
			for shuff in np.arange(nshuff_e+1,nshuff2+1):
				Age,Sex = shuff_demo(Age,Sex)
				subh = even_out(Age,Sex)
				ISC_w,_ = ISC_w_calc(D,n_vox,n_time,nsub,subh)
				roidict[task]['ISC_e'][shuff] = ISC_w[0] - ISC_w[1]
		if g_p < 0.05 and nshuff_g<nshuff2:
			print('g_diff')
			D,Age,Sex = load_D(roil,task,bins)
			nsub_,n_vox,n_time = D.shape
			nsub=nsub_//2
			roidict[task]['ISC_g'] = np.append(taskv['ISC_g'], np.zeros((nshuff2-nshuff_g,n_vox)), axis=0)
			roidict[task]['ISC_g_time'] = np.append(taskv['ISC_g_time'], np.zeros((nshuff2-nshuff_g,n_vox,n_time)), axis=0)
			for shuff in np.arange(nshuff_g+1,nshuff2+1):
				Age,Sex = shuff_demo(Age,Sex)
				subh = even_out(Age,Sex)
				ISC_w,groups = ISC_w_calc(D,n_vox,n_time,nsub,subh)
				ISC_b_time = []
				for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						ISC_b_time.append(np.multiply(groups[0,htmp1], groups[1,htmp2]))
				denom = np.sqrt(ISC_w[0]) * np.sqrt(ISC_w[1])
				roidict[task]['ISC_g_time'][shuff] = np.sum(ISC_b_time, axis=0)/4/np.tile(denom,(n_time,1)).T
				roidict[task]['ISC_g'][shuff] = np.sum([np.sum(i,axis=1)/(n_time-1) for i in ISC_b_time], axis=0)/4/denom
	dd.io.save(roi,roidict)
				
			
	
