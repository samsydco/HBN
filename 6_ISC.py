#!/usr/bin/env python3

import os
import glob
import tqdm
import numpy as np
import deepdish as dd
from scipy.stats import zscore, pearsonr
from HMM_settings import *

bins = [0,4]
nbins = len(bins)
seeds = [f[-1] for f in glob.glob(roidir+'*')]
savedir = ISCpath+'shuff_Yeo_outlier_'
nsub = 40
nshuff2perm=1000
task = 'DM'
n_time = 750

for seed in tqdm.tqdm(seeds):
	for roi in glob.glob(roidir+seed+'/'+'*.h5'):
		roi_short = roi.split('/')[-1][:-3]
		if not os.path.exists(savedir+seed): os.makedirs(savedir+seed)
		savef = savedir+seed+'/'+roi_short+'.h5'
		roidict = {task:{}}
		vall = dd.io.load(roi,'/vall')
		n_vox = len(vall)
		D,Age,Sex = load_D(roi,task,bins)
		shuffl = []
		if os.path.exists(savef):
			e_p,nshuff_ = p_calc(dd.io.load(savef,'/'+task+'/ISC_e'),'e')
			g_p,nshuff_ = p_calc(dd.io.load(savef,'/'+task+'/ISC_g'),'g')
			nshuff2 = nshuff2perm + nshuff_
			nshuff_all = 0
			e_p_all = g_p_all = 1
			if os.path.exists(pvals_file):
				nshuff_all = len(dd.io.load(pvals_file, '/roidict/'+roi_short+'/ISC_e/shuff'))
				e_p_all = dd.io.load(pvals_file, '/roidict/'+roi_short+'/ISC_e/p')
				g_p_all = dd.io.load(pvals_file, '/roidict/'+roi_short+'/ISC_g/p')
				if nshuff_all > nshuff2: nshuff2 = nshuff_all
			if ((e_p < 0.05 or g_p < 0.05) and nshuff_<nshuff2perm) or (e_p == 0 or g_p == 0) or (nshuff_ < nshuff_all and (e_p_all < 0.05 or g_p_all < 0.05)):
				roidict[task]['ISC_w'] = np.append(dd.io.load(savef,'/'+task+'/ISC_w'), np.zeros((nshuff2-nshuff_,nbins,n_vox)), axis=0)
				roidict[task]['ISC_e'] = np.append(dd.io.load(savef,'/'+task+'/ISC_e'), np.zeros((nshuff2-nshuff_,n_vox)), axis=0)
				roidict[task]['ISC_b'] = np.append(dd.io.load(savef,'/'+task+'/ISC_b'), np.zeros((nshuff2-nshuff_,4,n_vox)), axis=0)
				roidict[task]['ISC_g'] = np.append(dd.io.load(savef,'/'+task+'/ISC_g'), np.zeros((nshuff2-nshuff_,n_vox)), axis=0)
				# memory errors occur if try to load all at once:
				ISC_g_time = np.zeros((nshuff_+1,n_vox,n_time),dtype='float16')
				for shuff in range(nshuff_+1):
					ISC_g_time[shuff] = dd.io.load(savef,'/'+task+'/ISC_g_time',sel=dd.aslice[shuff,:,:])
				roidict[task]['ISC_g_time'] = np.append(ISC_g_time, np.zeros((nshuff2-nshuff_,n_vox,n_time),dtype='float16'), axis=0)
				shuffl = np.arange(nshuff_+1,nshuff2+1)
		else:
			roidict[task] = {'vall':vall, 'ISC_w':np.zeros((nshuff+1,nbins,n_vox)), 'ISC_b':np.zeros((nshuff+1,4,n_vox)), 'ISC_g_time':np.zeros((nshuff+1,n_vox,n_time)), 'ISC_g':np.zeros((nshuff+1,n_vox)), 'ISC_e':np.zeros((nshuff+1,n_vox))}
			shuffl = np.arange(nshuff+1)
		for shuff in shuffl:
			if shuff !=0:
				Age,Sex = shuff_demo(shuff,Age,Sex)
			subh = even_out(Age,Sex)
			roidict[task]['ISC_w'][shuff], groups =\
				ISC_w_calc(D,n_vox,n_time,nsub,subh)
			roidict[task]['ISC_e'][shuff] = roidict[task]['ISC_w'][shuff,0] -\
											roidict[task]['ISC_w'][shuff,1]
			ISC_b_time = []
			idx = 0
			for htmp1 in [0,1]:
				for htmp2 in [0,1]:
					ISC_b_time.append(np.multiply(groups[0,htmp1], groups[1,htmp2]))
					roidict[task]['ISC_b'][shuff,idx] = np.sum(ISC_b_time[-1], axis=1)/(n_time-1)
					idx += 1
			denom = np.sqrt(roidict[task]['ISC_w'][shuff,0]) * \
					np.sqrt(roidict[task]['ISC_w'][shuff,1])
			roidict[task]['ISC_g_time'][shuff] = np.sum(ISC_b_time, axis=0)/4/np.tile(denom,(n_time,1)).T
			roidict[task]['ISC_g'][shuff] = np.sum(roidict[task]['ISC_b'][shuff], axis=0)/4/denom
		if len(shuffl)>0:
			dd.io.save(savef,roidict)

														 
														 