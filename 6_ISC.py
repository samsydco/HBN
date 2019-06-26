#!/usr/bin/env python3

import h5py
import os
import glob
import tqdm
import numpy as np
import deepdish as dd
import pandas as pd
from datetime import datetime, date
from scipy.stats import zscore
from scipy.spatial.distance import squareform
from settings import *
from ISC_settings import *

# check if last ISC file has as many subjects as subord
cond = len(subord) != dd.io.load(ISCpath+'ISC_'+str(min([datetime.strptime(i.split('/')[-1].split('.h5')[0].split('ISC_')[1],'%Y-%m-%d') for i in glob.glob(ISCpath+'ISC_*') if '2019' in i], key=lambda x: abs(x - datetime.today()))).split(' ')[0]+'.h5',['/TP/data/0'])[0].shape[-1]
# where ISC_data.h5 is created in 5_AudioCorr.py
cond = len(subord) != dd.io.load(ISCpath+'ISC_data.h5',['/TP/0'])[0].shape[-1]

ISCf = 'ISC_'+str(date.today())+'.h5'
if os.path.exists(ISCpath+ISCf):
    os.remove(ISCpath+ISCf)

for task in ['DM','TP']:
	print(task)
	sh = dd.io.load(subord[0],['/'+task+'/L'])[0].shape
	# check if last ISC file has as many subjects as subord
	if cond:
		D = np.empty((len(subord),sh[0]*2,sh[1]),dtype='float16')
		for s, sub in tqdm.tqdm(enumerate(subord)):
			D[s,:,:] = np.concatenate([dd.io.load(sub,['/'+task+'/L'])[0], dd.io.load(sub,['/'+task+'/R'])[0]], axis=0)
		D = np.transpose(D,(1,2,0))
	else:
		D = np.empty((sh[0]*2,sh[1],len(subord)),dtype='float16')
		Dtmp = dd.io.load(ISCpath+'ISC_'+str(min([datetime.strptime(i.split('/')[-1].split('.h5')[0].split('ISC_')[1],'%Y-%m-%d') for i in glob.glob(ISCpath+'ISC_*') if '2019' in i], key=lambda x: abs(x - datetime.today()))).split(' ')[0]+'.h5',['/'+task+'/data/'])[0]
		for key, value in Dtmp.items():
			D[:,0+250*int(key):250+250*int(key),:] = value
	n_vox,n_time,n_subj=D.shape
	with h5py.File(ISCpath+ISCf) as hf:
		grp = hf.create_group(task)
		if cond:
			grpd = grp.create_group('data')
			for i in range(int(n_time/250)):
				grpd.create_dataset(str(i), data=D[:,0+250*i:250+250*i,:])
		for k,v in phenol.items():
			if 'pc' in k:
				print(k)
				ISC = np.zeros((n_vox,n_subj),dtype='float16')
				# Loop across choice of leave-one-out subject
				for loo_subj in tqdm.tqdm(range(n_subj)):
					bv = v[loo_subj]
					group = np.zeros((n_vox,n_time),dtype='float16')
					groupn = np.ones((n_vox,n_time),dtype='int')*sum([bv==vi for vi in v])-1
					for i in range(n_subj):
						if i != loo_subj and v[i] == bv:
							group = np.nansum(np.stack((group,D[:,:,i])),axis=0)
							verts = np.argwhere(np.isnan(D[:,:,i]))
							groupn[verts[:, 0],verts[:,1]] = groupn[verts[:,0],verts[:,1]]-1
					group = zscore(group/groupn,axis=1)
					subj = zscore(D[:, :, loo_subj],axis=1)
					ISC[:,loo_subj] = np.sum(np.multiply(group,subj),axis=1)/(n_time-1)
				grp.create_dataset('ISC_persubj_'+k, data=ISC)

with h5py.File(ISCpath+ISCf) as hf:
	string_dt = h5py.special_dtype(vlen=str)
	hf.create_dataset("subord",data=np.array(subord).astype('|S71'),dtype=string_dt)

'''
    #pairs = list(itertools.combinations(range(len(subord)), 2))
    voxel_iscs = []
    for v in np.arange(data.shape[0]):
        voxel_data = data[v, :, :].T
        iscs = squareform(np.corrcoef(voxel_data), checks=False)
        # iscs[int(comb(10,2)-comb(10-i,2) + (j-i-1))] 
        voxel_iscs.append(iscs)
    iscs = np.column_stack(voxel_iscs)
'''
