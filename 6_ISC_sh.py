#!/usr/bin/env python3

import tqdm
import numpy as np
import deepdish as dd
from datetime import datetime, date
from scipy.stats import zscore
from scipy.spatial.distance import squareform
from settings import *
from ISC_settings import *
del phenol['all']
phenolperm = phenol

start_date = str(date.today())
ISCfs = ISCpath+'shuff/ISC_'+start_date+'_'
nsh = 1 #5 split-half iterations

def ISCe_calc(fstr,cond):
	ISCe = dd.io.load(fstr,'/'+cond+'/'+'ISC_w')[0]-dd.io.load(fstr,'/'+cond+'/'+'ISC_w')[1]
	return ISCe
def ISCg_calc(fstr,cond):
    ISCg = sum(dd.io.load(fstr,'/'+cond+'/'+'ISC_b'))/4/(np.sqrt(dd.io.load(fstr,'/'+cond+'/'+'ISC_w')[0])*
				   np.sqrt(dd.io.load(fstr,'/'+cond+'/'+'ISC_w')[1]))
    return ISCg

def shuff_check(fstr,cond,nshuff,n_vox):
	fstrtmp = ('_').join(fstr.split('_')[:-1])+'_'+str(0)+'.h5'
	ISCg = ISCg_calc(fstrtmp,cond)
	ISCe = ISCe_calc(fstrtmp,cond)
	vvecte = np.zeros((nshuff,len(ISCe)))
	vvectg = np.zeros((nshuff,len(ISCg)))
	goodvtmp = dd.io.load(('_').join(fstr.split('_')[:-1])+'_'+str(nshuff)+'.h5','/'+cond+'/'+'good_v_indexes')
	for shuff in np.arange(1,nshuff+1):
		fstrtmp = ('_').join(fstr.split('_')[:-1])+'_'+str(shuff)+'.h5'
		vvecte[shuff-1,goodvtmp] = ISCe_calc(fstrtmp,cond)
		vvectg[shuff-1,goodvtmp] = ISCg_calc(fstrtmp,cond)
	vertsg = np.asarray([np.sum(vvectg[:,v]<ISCg[v])/nshuff for v in range(len(ISCg))]) #(previously: np.sum(vvect<ISC[v]))
	vertse = np.asarray([np.sum(abs(vvecte[:,v])>abs(ISCe[v]))/nshuff for v in range(len(ISCe))])
	vertsidxe = [i for i, v in enumerate(vertse) if v<0.1 and ~np.isnan(ISCe[i]) and                            vvecte[-1,i]!=0]
	vertsidxg = [i for i, v in enumerate(vertsg) if v<0.1 and ~np.isnan(ISCg[i]) and                            vvectg[-1,i]!=0]
	good_v_indexes = list(set(vertsidxe+vertsidxg))
	return good_v_indexes


nshuff = 10000 # number of shuffles
for s in range(nsh):
	for task in ['DM','TP']:
		phenol = phenolperm
		sh = dd.io.load(subord[0],['/'+task+'/L'])[0].shape
		#D = np.empty((len(subord),sh[0]*2,sh[1]),dtype='float16') # UGH memory error :(
		#for sidx, sub in tqdm.tqdm(enumerate(subord)):
		#	D[sidx,:,:] = np.concatenate([dd.io.load(sub,['/'+task+'/L'])[0], dd.io.load(sub,['/'+task+'/R'])[0]], axis=0)
		#D = np.transpose(D,(1,2,0))
		n_vox=sh[0]*2
		n_time = sh[1]
		n_subj = len(subord)
		#n_vox,n_time,n_subj=D.shape
		good_v_indexes = {key: np.arange(n_vox) for key in phenol.keys()}
		for shuff in tqdm.tqdm(range(nshuff+1)):
			fstr = ISCfs+task+str(s)+'_shuff_'+str(shuff)+'.h5'
			shuffdict = {k:{'subs':[],'ISC_w':[],'ISC_b':[],
							'good_v_indexes':good_v_indexes[k]} for k in ['age','sex']} #phenol.keys()}
			for k in shuffdict.keys():
				n_vox = len(good_v_indexes[k])
				v2 = phenolperm['sex'] if k!='sex' else phenolperm['age']
				shuffdict[k]['subs'] = even_out(phenol[k],v2)
				groups = np.zeros((2,2,n_vox,n_time),dtype='float16')
				for h in [0,1]: # split all or between T / F
					for htmp in [0,1]: # split within split
						group = np.zeros((n_vox,n_time),dtype='float16')
						groupn = np.ones((n_vox,n_time),dtype='int')*len(shuffdict[k]['subs'][h][htmp])
						for i in shuffdict[k]['subs'][h][htmp]: # mem error in next line (91 reps):
							D = np.concatenate([dd.io.load(subord[i],['/'+task+'/L'])[0], dd.io.load(subord[i],['/'+task+'/R'])[0]], axis=0)
							group = np.nansum(np.stack((group,D[good_v_indexes[k],:])),axis=0)
							nanverts = np.argwhere(np.isnan(D[good_v_indexes[k],:]))
							groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
						groups[h,htmp] = zscore(group/groupn,axis=1)
					shuffdict[k]['ISC_w'].append(np.sum(np.multiply(groups[h,0],groups[h,1]),axis=1)/(n_time-1))
				for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						shuffdict[k]['ISC_b'].append(np.sum(np.multiply(groups[0,htmp1],groups[1,htmp2]),axis=1)/(n_time-1)) # correlate across groups
				if any(shu==shuff for shu in [101,1001]):
					good_v_indexes[k] = shuff_check(fstr,k,shuff-1,n_vox)
					print('The number of verts left after',str(shuff),\
						  'iterations for group',k,'is',len(good_v_indexes[k]))
			dd.io.save(fstr,shuffdict)
				
			# randomly shuffle phenol:
			for k,v in phenol.items():
				nonnanidx = np.argwhere(~np.isnan(phenol[k]))
				randidx = np.random.permutation(nonnanidx)
				phenol[k] = [v[randidx[nonnanidx==idx][0]] if idx in nonnanidx else i for idx,i in enumerate(v)]
		
		