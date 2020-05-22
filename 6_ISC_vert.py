#!/usr/bin/env python3

import glob
import tqdm
import random
import numpy as np
import deepdish as dd
from scipy.stats import zscore
from ISC_settings import *

nTR=[750,250]
bins = [0,4]
nvox = 81924
nsub = 41
ISCfs = ISCpath+'shuff_bin/ISC_'

def ISCe_calc(ISC_w):
	ISCe = ISC_w[1]-ISC_w[0]
	return ISCe
def ISCg_calc(ISC_w,ISC_b):
    ISCg = np.sum(np.stack(ISC_b),axis=0)/4/(np.sqrt(ISC_w[0])*np.sqrt(ISC_w[1]))
    return ISCg


for ti,task in enumerate(['DM','TP']):
	print(task)
	n_time = nTR[ti]
	D = np.empty((nsub*2,nvox,n_time),dtype='float16')
	Age = []
	Sex = []
	for bi,b in enumerate(bins):
		subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
		Sex.extend([Phenodf['Sex'][Phenodf['EID'] == shortsub(sub)].iloc[0] for sub in subl])
		Age.extend([bi]*len(subl))
		sub_ = 0 if b==0 else nsub # young and then old in D
		for sidx, sub in enumerate(subl):
			D[sidx+sub_] = np.concatenate([dd.io.load(sub,['/'+task+'/L'])[0], \
										   dd.io.load(sub,['/'+task+'/R'])[0]], axis=0)
	Ageperm = Age
	for shuff in tqdm.tqdm(range(nshuff+1)):
		ISC_w = []
		ISC_b = []
		subh = even_out(Age,Sex)
		groups = np.zeros((2,2,nvox,n_time),dtype='float16')
		for h in [0,1]: # split all or between T / F
			for htmp in [0,1]: # split within split
				group = np.zeros((nvox,n_time),dtype='float16')
				groupn = np.ones((nvox,n_time),dtype='int')*len(subh[h][htmp])
				for i in subh[h][htmp]:
					group = np.nansum(np.stack((group,D[i])),axis=0)
					nanverts = np.argwhere(np.isnan(D[i]))
					groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
				groups[h,htmp] = zscore(group/groupn,axis=1)
			ISC_w.append(np.sum(np.multiply(groups[h,0],groups[h,1]),axis=1)/(n_time-1))
		for htmp1 in [0,1]:
			for htmp2 in [0,1]:
				ISC_b.append(np.sum(np.multiply(groups[0,htmp1],\
												groups[1,htmp2]),axis=1)/(n_time-1)) # correlate across groups
		e_diff = ISCe_calc(ISC_w)
		g_diff = ISCg_calc(ISC_w,ISC_b)
		dd.io.save(ISCfs+'_'.join([task,str(shuff)]),{'ISC_w':ISC_w,'ISC_b':ISC_b,\
													 'e_diff':e_diff,'g_diff':g_diff})
		# Now shuffle Age:
		random.shuffle(Age)


