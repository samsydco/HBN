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
from datetime import datetime, date
from scipy.stats import zscore
from scipy.spatial.distance import squareform
from settings import *
from ISC_settings import *

ISCf = 'ISC_'+str(date.today())+'.h5'
if os.path.exists(ISCpath+ISCf):
    os.remove(ISCpath+ISCf)
	
n_subj = 50 # must be even
# find minimum number of subjects per age bin
bindist = [min([agedist[1][0][0][b],agedist[0][0][0][b]]) for b in range(nbins)]
# indexes of bins to draw from in final sample:
bindist = [np.random.choice(nbins,p=nsubbin/sum(nsubbin)) for n in range(n_subj//2)]
subl = [agedist[i][1][n][np.random.choice(len(agedist[i][1][n]))] for n in bindist for i in [0,1]]
dups = np.unique([x for x in subl if subl.count(x) > 1])
for d in dups:
	di = [i for i in [0,1] if any(d in sl for sl in agedist[i][1])][0]
	dloc = [idx for idx,s in enumerate(agedist[di][1]) if d in s][0]
	for didx in [i for i,s in enumerate(subl) if s == d][1:]:
		while True:
			sn = agedist[di][1][dloc][np.random.choice(len(agedist[di][1][dloc]))]
			if sn in subl:
				continue
			else:
				break
		subl[didx] = sn
dd.io.save(ISCpath+ISCf,{'subs':subl})

for task in ['DM','TP']:
	print(task)
	sh = dd.io.load(subl[0],['/'+task+'/L'])[0].shape
	# check if last ISC file has as many subjects as subord
	D = np.empty((len(subl),sh[0]*2,sh[1]),dtype='float16')
	for s, sub in tqdm.tqdm(enumerate(subl)):
		D[s,:,:] = np.concatenate([dd.io.load(sub,['/'+task+'/L'])[0], dd.io.load(sub,['/'+task+'/R'])[0]], axis=0)
	D = np.transpose(D,(1,2,0))
	n_vox,n_time,n_subj=D.shape
	with h5py.File(ISCpath+ISCf) as hf:
		grp = hf.create_group(task)
		grpd = grp.create_group('data')
		for i in range(int(n_time/250)):
			grpd.create_dataset(str(i), data=D[:,0+250*i:250+250*i,:])
		voxel_iscs = []
		for v in tqdm.tqdm(np.arange(n_vox)):
			voxel_data = D[v].T
			# Correlation matrix for all pairs of subjects (triangle)
			iscs = squareform(np.corrcoef(voxel_data), checks=False)
			voxel_iscs.append(iscs)
		iscs = np.column_stack(voxel_iscs)
		grp.create_dataset('pairwise_isc', data=iscs)
	