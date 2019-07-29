#!/usr/bin/env python3

import os
import h5py
import tqdm
import numpy as np
import deepdish as dd
from random import shuffle
from datetime import datetime, date
from scipy.stats import zscore
from scipy.spatial.distance import squareform
from settings import *
from ISC_settings import *
phenolperm = phenol

def rand_shuff(v):
	# randomly shuffle value in phenol:
	nonnanidx = np.argwhere(~np.isnan(v))
	randidx = np.random.permutation(nonnanidx)
	v = [v[randidx[nonnanidx==idx][0]] if idx in nonnanidx else i for idx,i in enumerate(v)]
	return v

ISCf = 'ISC_'+str(date.today())+'_sh_2.h5'
if os.path.exists(ISCpath+ISCf):
    os.remove(ISCpath+ISCf)
dd.io.save(ISCpath+ISCf,{'subs':subord,'ages':agel,'phenodict':phenol,'pcs':pcl})
nsh = 1 #5 split-half iterations
nshuff = 100 # number of shuffles
for s in range(nsh):
	for task in ['DM','TP']:
		print('sh = ',s,task)
		sh = dd.io.load(subord[0],['/'+task+'/L'])[0].shape
		D = np.empty((len(subord),sh[0]*2,sh[1]),dtype='float16')
		for sidx, sub in tqdm.tqdm(enumerate(subord)):
			D[sidx,:,:] = np.concatenate([dd.io.load(sub,['/'+task+'/L'])[0], dd.io.load(sub,['/'+task+'/R'])[0]], axis=0)
		D = np.transpose(D,(1,2,0))
		n_vox,n_time,n_subj=D.shape
		with h5py.File(ISCpath+ISCf) as hf:
			grp = hf.create_group(task+str(s))
			for k,v in phenol.items():
				ISC_SH_w = np.zeros((2,n_vox,nshuff+1))
				ISC_SH_b = np.zeros((2,2,n_vox,nshuff+1))
				for shuff in tqdm.tqdm(range(nshuff+1)):
					v2 = phenolperm['sex'] if k!='sex' else phenolperm['age']
					subh = even_out(v,v2)
					groups = np.zeros((2,2,n_vox,n_time),dtype='float16')
					for h in [0,1]: # split all or between T / F
						for htmp in [0,1]: # split within split
							group = np.zeros((n_vox,n_time),dtype='float16')
							groupn = np.ones((n_vox,n_time),dtype='int')*n_subj
							for i in subh[h][htmp]:
								group = np.nansum(np.stack((group,D[:,:,i])),axis=0)
								nanverts = np.argwhere(np.isnan(D[:,:,i]))
								groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
							groups[h,htmp] = zscore(group/groupn,axis=1)
						ISC_SH_w[h,:,shuff]=\
						np.sum(np.multiply(groups[h,0],groups[h,1]),axis=1)/(n_time-1)
					for htmp1 in [0,1]:
						for htmp2 in [0,1]:
							ISC_SH_b[htmp1,htmp2,:,shuff]=\
							np.sum(np.multiply(groups[0,htmp1],\
											   groups[1,htmp2]),axis=1)/(n_time-1) # correlate across groups
					v = rand_shuff(v)
				grp.create_dataset('ISC_SH_w_'+k,data=ISC_SH_w)
				grp.create_dataset('ISC_SH_b_'+k,data=ISC_SH_b)
				
		
		