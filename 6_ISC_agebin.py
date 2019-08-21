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

ISCf = 'ISC_'+str(date.today())+'_age.h5'
if os.path.exists(ISCpath+ISCf):
    os.remove(ISCpath+ISCf)

for shuff in tqdm.tqdm(range(nshuff+1)):
	with h5py.File(ISCpath+ISCf) as hf:
		grps = hf.create_group('shuff_'+str(shuff))
		ageeq,lenageeq,minageeq = binagesubs(agel,phenol['sex'],eqbins,subord)
		for b in range(nbinseq):
			subhtmp = even_out([True]*(lenageeq[0][b]+lenageeq[1][b]),[False]*minageeq[0]+[True]*minageeq[1])
			subh = [[[], []] for i in range(2)]
			for htmp in [0,1]:
				subh[0][htmp] = [s for s in subhtmp[htmp][0]+subhtmp[htmp][1] if s<minageeq[0]]
				subh[1][htmp] = [s for s in subhtmp[htmp][0]+subhtmp[htmp][1] if s>=minageeq[0]]
			subl = [k for i in [0,1] for k in [ageeq[i][1][b][idx] for idx in   np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]]
			grpb = grps.create_group('bin_'+str(b))
			grpb.create_dataset('subl', (len(subl),1),'S48', [n.encode("ascii", "ignore") for n in subl])
			grpb.create_dataset('subh',data=subh)
			for task in ['DM','TP']:
				grp = grpb.create_group(task)
				sh = dd.io.load(subl[0],['/'+task+'/L'])[0].shape
				D = np.empty((len(subl),sh[0]*2,sh[1]),dtype='float16')
				for sidx, sub in enumerate(subl):
					D[sidx,:,:] = np.concatenate([dd.io.load(sub,['/'+task+'/L'])[0], dd.io.load(sub,['/'+task+'/R'])[0]], axis=0)
				D = np.transpose(D,(1,2,0))
				n_vox,n_time,n_subj=D.shape
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
					grp.create_dataset('ISC_SH_w_'+str(h),\
						data=np.sum(np.multiply(groups[h,0],groups[h,1]),axis=1)/(n_time-1))
				for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						grp.create_dataset\
						('ISC_SH_b_'+str(htmp1)+'_'+str(htmp2),\
						data=np.sum(np.multiply(groups[0,htmp1],\
												groups[1,htmp2]),axis=1)/(n_time-1)) # correlate across groups
										
	# randomly shuffle ages:
	ageidx = np.random.permutation(len(agel))
	agel = [agel[ageidx[idx]] for idx,age in enumerate(agel)]
	phenol['sex'] = [phenol['sex'][ageidx[idx]] for idx,age in enumerate(phenol['sex'])]
		
		


