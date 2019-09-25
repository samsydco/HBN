#!/usr/bin/env python3

# Look at ISC change with age!!

import os
import h5py
import tqdm
import itertools
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
		subla = []
		for b in range(nbinseq):
			grpb = grps.create_group('bin_'+str(b))
			subl = [[],[]]
			for i in [0,1]:
				subg = [ageeq[i][1][b][idx] for idx in np.random.choice(lenageeq[i][b],divmod(minageeq[i],2)[0]*2,replace=False)]
				subl[0].extend(subg[:divmod(minageeq[i],2)[0]])
				subl[1].extend(subg[divmod(minageeq[i],2)[0]:])
			subla.append(subl)
			for h in [0,1]:
				grpb.create_dataset('subl_'+str(h), (len(subl[h]),1),'S48', [n.encode("ascii", "ignore") for n in subl[h]])
		n_subj = len(subla[0][0])*2
		for task in ['DM','TP']:
			grp = grps.create_group(task)
			n_time = dd.io.load(subl[0][0],['/'+task+'/L'])[0].shape[1]
			groups = np.zeros((nbinseq,2,n_vox,n_time),dtype='float16')
			for b in range(nbinseq):
				grpb = grp.create_group('bin_'+str(b))
				for h in [0,1]: # split all or between T / F
					group = np.zeros((n_vox,n_time),dtype='float16')
					groupn = np.ones((n_vox,n_time),dtype='int')*n_subj
					for sub in subla[b][h]:
						d = np.concatenate([dd.io.load(sub,['/'+task+'/L'])[0], dd.io.load(sub,['/'+task+'/R'])[0]], axis=0)
						group = np.nansum(np.stack((group,d)),axis=0)
						nanverts = np.argwhere(np.isnan(d))
						groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
					groups[b,h] = zscore(group/groupn,axis=1)
				grpb.create_dataset('ISC_w',data=np.sum(np.multiply(\
					groups[b,0],groups[b,1]),axis=1)/(n_time-1))
			for p in itertools.combinations(range(nbinseq),2):
				grpp = grp.create_group('bin_'+str(p[0])+'_'+str(p[1]))
				for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						grpp.create_dataset(str(htmp1)+'_'+str(htmp2),\
						data=np.sum(np.multiply(groups[p[0],htmp1],groups[p[1],htmp2]),axis=1)/(n_time-1)) # correlate across bins
	# randomly shuffle ages:
	ageidx = np.random.permutation(len(agel))
	agel = [agel[ageidx[idx]] for idx,age in enumerate(agel)]
	phenol['sex'] = [phenol['sex'][ageidx[idx]] for idx,age in enumerate(phenol['sex'])]
		
		


