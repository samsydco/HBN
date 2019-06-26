#!/usr/bin/env python3

# Pairwise correlation with 50 subjs
# 1a) Old vs Young, equal M and F in both
# 1b) Old ~ Young, and Young ~ Old
# 2a) M vs F, equal Old and Young in both
# 2b) M ~ F, and F ~ M

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

ISCf = 'ISC_'+str(date.today())+'_sh_2.h5'
if os.path.exists(ISCpath+ISCf):
    os.remove(ISCpath+ISCf)
dd.io.save(ISCpath+ISCf,{'subs':subord,'ages':agel,'phenodict':phenol,'pcs':pcl})
nsh = 1 #5 split-half iterations
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
				print(k)
				v2 = phenol['sex'] if k!='sex' else phenol['age']
				subh = even_out(v,v2)
				grp.create_dataset('subs_'+k,data=subh)
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
					grp.create_dataset('ISC_SH_w_'+k+'_'+str(h),\
						data=np.sum(np.multiply(groups[h,0],groups[h,1]),axis=1)/(n_time-1))
				for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						grp.create_dataset('ISC_SH_b_'+k+'_'+str(htmp1)+'_'+str(htmp2),\
					data=np.sum(np.multiply(groups[0,htmp1],groups[1,htmp2]),axis=1)/(n_time-1)) # correlate across groups
		
		