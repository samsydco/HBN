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
del phenol['all']
phenolperm = phenol

ISCf = ISCpath+'ISC_'+str(date.today())+'_sh_2.h5'
if os.path.exists(ISCf):
    os.remove(ISCf)
dd.io.save(ISCf,{'subs':subord,'ages':agel,'phenodict':phenol,'pcs':pcl})
nsh = 1 #5 split-half iterations

def ISCe_calc(iscf,task,cond,sh,shuff):
	ISCe = dd.io.load(iscf,'/'+task+str(sh)+'/shuff_'+shuff+'/'+cond+'/'+'ISC_SH_w/0') - \
		   dd.io.load(iscf,'/'+task+str(sh)+'/shuff_'+shuff+'/'+cond+'/'+'ISC_SH_w/1')
	return ISCe
def ISCg_calc(iscf,task,cond,sh,shuff):
	ls = '/'+task+str(sh)+'/shuff_'+shuff+'/'+cond+'/'+'ISC_SH_b/'
	ISCg = np.zeros(81924) 
	for i in dd.io.load(iscf,ls).keys():
		ISCg += dd.io.load(iscf,ls+i)
	ISCg = ISCg/4/(np.sqrt(dd.io.load(iscf,'/'+task+str(sh)+'/shuff_'+shuff+'/'+cond+'/'+'ISC_SH_w/0'))\
				  *np.sqrt(dd.io.load(iscf,'/'+task+str(sh)+'/shuff_'+shuff+'/'+cond+'/'+'ISC_SH_w/1')))
	return ISCg

def shuff_check(iscf,task,cond,sh,nshuff):
	ISCg = ISCg_calc(iscf,task,cond,sh,str(0))
	ISCe = ISCe_calc(iscf,task,cond,sh,str(0))

	vvecte = np.zeros((nshuff,len(ISCe)))
	vvectg = np.zeros((nshuff,len(ISCg)))
	for shuff in np.arange(1,nshuff+1):
		vvecte[shuff-1] = ISCe_calc(iscf,task,cond,sh,str(shuff))
		vvectg[shuff-1] = ISCg_calc(iscf,task,cond,sh,str(shuff))
	vertsg = np.asarray([np.sum(vvectg[:,v]<ISCg[v])/nshuff for v in range(len(ISCg))]) #(previously: np.sum(vvect<ISC[v]))
	vertse = np.asarray([np.sum(abs(vvecte[:,v])>abs(ISCe[v]))/nshuff for v in range(len(ISCe))])
	vertsidxe = [i for i, v in enumerate(vertse) if v<0.1 and ~np.isnan(ISCe[i])]
	vertsidxg = [i for i, v in enumerate(vertsg) if v<0.1 and ~np.isnan(ISCg[i])]
	good_v_indexes = list(set(vertsidxe+vertsidxg))
	return good_v_indexes


nshuff = 10000 # number of shuffles
for s in range(nsh):
	for task in ['DM','TP']:
		phenol = phenolperm
		print('sh = ',s,task)
		sh = dd.io.load(subord[0],['/'+task+'/L'])[0].shape
		D = np.empty((len(subord),sh[0]*2,sh[1]),dtype='float16')
		for sidx, sub in tqdm.tqdm(enumerate(subord)):
			D[sidx,:,:] = np.concatenate([dd.io.load(sub,['/'+task+'/L'])[0], dd.io.load(sub,['/'+task+'/R'])[0]], axis=0)
		D = np.transpose(D,(1,2,0))
		n_vox,n_time,n_subj=D.shape
		with h5py.File(ISCf,'a') as hf:
			grpt = hf.create_group(task+str(s))
			good_v_indexes = {key: np.arange(81924) for key in phenol.keys()}
			for shuff in tqdm.tqdm(range(nshuff+1)):
				grps = grpt.require_group('shuff_'+str(shuff))
				for k,v in phenol.items():
					grpk = grps.require_group(k)
					n_vox = len(good_v_indexes[k])
					v2 = phenolperm['sex'] if k!='sex' else phenolperm['age']
					subh = even_out(v,v2)
					grpk.create_dataset('subs',data=subh)
					groups = np.zeros((2,2,n_vox,n_time),dtype='float16')
					for h in [0,1]: # split all or between T / F
						for htmp in [0,1]: # split within split
							group = np.zeros((n_vox,n_time),dtype='float16')
							groupn = np.ones((n_vox,n_time),dtype='int')*n_subj
							for i in subh[h][htmp]: # mem error in next line (91 reps):
								group = np.nansum(np.stack((group,D[good_v_indexes[k],:,i])),axis=0)
								nanverts = np.argwhere(np.isnan(D[good_v_indexes[k],:,i]))
								groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
							groups[h,htmp] = zscore(group/groupn,axis=1)
						grpw = grpk.require_group('ISC_SH_w')
						grpw.create_dataset(str(h),\
						data=np.sum(np.multiply(groups[h,0],groups[h,1]),axis=1)/(n_time-1))
					for htmp1 in [0,1]:
						for htmp2 in [0,1]:
							grpb = grpk.require_group('ISC_SH_b')
							grpb.create_dataset(str(htmp1)+'_'+str(htmp2),\
							data=np.sum(np.multiply(groups[0,htmp1],groups[1,htmp2]),axis=1)/(n_time-1)) # correlate across groups
					if any(shu==shuff for shu in [101,1001]):
						good_v_indexes[k] = shuff_check(ISCf,task,k,s,shuff-1)
						print('The number of verts left after',str(shuff),\
							  'iterations for group',k,'is',len(good_v_indexes[k]))
						grpk.create_dataset('good_v_indexes',data=good_v_indexes[k])
				
				# randomly shuffle phenol:
				for k,v in phenol.items():
					nonnanidx = np.argwhere(~np.isnan(phenol[k]))
					randidx = np.random.permutation(nonnanidx)
					phenol[k] = [v[randidx[nonnanidx==idx][0]] if idx in nonnanidx else i for idx,i in enumerate(v)]
		
		