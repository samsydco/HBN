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

ISCf = 'ISC_'+str(date.today())+'.h5'
if os.path.exists(ISCpath+ISCf):
    os.remove(ISCpath+ISCf)

def save2x2list(grpname,k,v):
	for h in [0,1]:
		for htmp in [0,1]:
			grpname.create_dataset(k+'_'+str(h)+'_'+str(htmp), data=v[h][htmp])

n_subj = np.sum(nsubbin,dtype=int) * 2
nsh = 5 #number of split-half iterations
for s in range(nsh):
	subl = [k for i in [0,1] for n in range(nbins)  if len(agedist[i][1][n]) > 0 for k in [agedist[i][1][n][idx] for idx in np.random.choice(len(agedist[i][1][n]),nsubbin[n],replace=False)]] # find minimum number of subjects per age bin
	shuffle(subl)
	agel,pcl,phenol = make_phenol(subl)
	splitidx,subdist,subsum = split_subj(phenol)						
	pheno_dict = {'subs':subl,'ages':agel,'phenodict':phenol,'pcs':pcl,'splitidx':splitidx,'subdist':subdist,'subsum':subsum}
	with h5py.File(ISCpath+ISCf) as hf:
		grps = hf.create_group('SH_'+str(s))
		grpp = grps.create_group('phenol')
		for k, v in pheno_dict.items():
			if type(v)==dict:
				grppp = grpp.create_group(k)
				for kk, vv in v.items():
					if len(vv)==2:
						save2x2list(grppp,kk,vv)
					elif type(vv)==dict:
						grpppp = grppp.create_group(kk)
						for kkk, vvv in vv.items():
							save2x2list(grpppp,kkk,vvv)
					else:
						grppp.create_dataset(kk, data=vv)
			elif type(v[0]) == str:
				grpp.create_dataset(k, (len(v),1),'S10', [n.encode("ascii", "ignore") for n in v])
			else:
				grpp.create_dataset(k, data=v)
		for task in ['DM','TP']:
			print('sh = ',s,task)
			sh = dd.io.load(subl[0],['/'+task+'/L'])[0].shape
			D = np.empty((len(subl),sh[0]*2,sh[1]),dtype='float16')
			for sidx, sub in tqdm.tqdm(enumerate(subl)):
				D[sidx,:,:] = np.concatenate([dd.io.load(sub,['/'+task+'/L'])[0], dd.io.load(sub,['/'+task+'/R'])[0]], axis=0)
			D = np.transpose(D,(1,2,0))
			n_vox,n_time,n_subj=D.shape
			grp = grps.create_group(task)
			#grpd = grp.create_group('data')
			#for i in range(int(n_time/250)):
			#	grpd.create_dataset(str(i), data=D[:,0+250*i:250+250*i,:])
			for k,v in phenol.items():
				print(k)
				groups = np.zeros((2,2,n_vox,n_time),dtype='float16')
				for h in [0,1]: # split all or between T / F
					for htmp in [0,1]: # split within split
						group = np.zeros((n_vox,n_time),dtype='float16')
						groupn = np.ones((n_vox,n_time),dtype='int')*n_subj
						for i in splitidx[k][h][htmp]:
							group = np.nansum(np.stack((group,D[:,:,i])),axis=0)
							nanverts = np.argwhere(np.isnan(D[:,:,i]))
							groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
						groups[h,htmp] = zscore(group/groupn,axis=1)
					grp.create_dataset('ISC_SH_w_'+k+'_'+str(h),\
						data=np.sum(np.multiply(groups[h,htmp],groups[h,htmp]),axis=1)/(n_time-1))
				for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						grp.create_dataset('ISC_SH_b_'+k+'_'+str(htmp1)+'_'+str(htmp2),\
					data=np.sum(np.multiply(groups[0,htmp1],groups[1,htmp2]),axis=1)/(n_time-1)) # correlate across groups
				
				

		
	