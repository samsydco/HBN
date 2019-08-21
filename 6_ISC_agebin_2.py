#!/usr/bin/env python3

# Look at ISC change with age!!

import os
import h5py
import tqdm
import numpy as np
import deepdish as dd
from datetime import date
from scipy.stats import zscore
from settings import *
from ISC_settings import *

n_vox = 81924

ISCf = 'ISC_'+str(date.today())+'_age_2.h5'
if os.path.exists(ISCpath+ISCf):
    os.remove(ISCpath+ISCf)

for shuff in tqdm.tqdm(range(nshuff+1)):
	with h5py.File(ISCpath+ISCf) as hf:
		grps = hf.create_group('shuff_'+str(shuff))
		ageeq,lenageeq,minageeq = binagesubs(agel,phenol['sex'],eqbins,subord)
		for b in range(nbinseq):
			grpb = grps.create_group('bin_'+str(b))
			subl = [[],[]]
			for i in [0,1]:
				subg = [ageeq[i][1][b][idx] for idx in np.random.choice(lenageeq[i][b],divmod(minageeq[i],2)[0]*2,replace=False)]
				subl[0].extend(subg[:divmod(minageeq[i],2)[0]])
				subl[1].extend(subg[divmod(minageeq[i],2)[0]:])
			for h in [0,1]:
				grpb.create_dataset('subl_'+str(h), (len(subl[h]),1),'S48', [n.encode("ascii", "ignore") for n in subl[h]])
			n_subj = len(subl[0])*2
			for task in ['DM','TP']:
				grp = grpb.create_group(task)
				n_time = dd.io.load(subl[0][0],['/'+task+'/L'])[0].shape[1]
				groups = np.zeros((2,n_vox,n_time),dtype='float16')
				for h in [0,1]: # split all or between T / F
					group = np.zeros((n_vox,n_time),dtype='float16')
					groupn = np.ones((n_vox,n_time),dtype='int')*n_subj
					for sub in subl[h]:
						d = np.concatenate([dd.io.load(sub,['/'+task+'/L'])[0], dd.io.load(sub,['/'+task+'/R'])[0]], axis=0)
						group = np.nansum(np.stack((group,d)),axis=0)
						nanverts = np.argwhere(np.isnan(d))
						groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
					groups[h] = zscore(group/groupn,axis=1)
				grp.create_dataset('ISC_SH',data=np.sum(np.multiply(\
					groups[0],groups[1]),axis=1)/(n_time-1))						
	# randomly shuffle ages:
	ageidx = np.random.permutation(len(agel))
	agel = [agel[ageidx[idx]] for idx,age in enumerate(agel)]
	phenol['sex'] = [phenol['sex'][ageidx[idx]] for idx,age in enumerate(phenol['sex'])]
		
		


