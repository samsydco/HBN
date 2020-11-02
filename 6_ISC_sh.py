#!/usr/bin/env python3

import glob
import tqdm
import numpy as np
import deepdish as dd
from scipy.stats import zscore
from settings import *
from ISC_settings import *

no_atypical = False
if no_atypical == True:
	dfd = pd.read_csv(metaphenopath+'Neurodevelopmental_Diagnosis_Frequency.csv')
	subord = [prepath+sub+'.h5' for sub in list(dfd.loc[dfd['No Diagnosis Given'] == True]['EID'])]
	agel,pcl,phenol = make_phenol(subord)
	ISCfs = ISCpath+'No_Diag/'
	nshuff = 0 # number of shuffles
else:
	ISCfs = ISCpath+'shuff/ISC_'
	nshuff = 100#0000 # number of shuffles
smallsub = False
if smallsub == True:
	dfd = pd.read_csv(metaphenopath+'Neurodevelopmental_Diagnosis_Frequency.csv')
	nsubs = len(dfd.loc[dfd['No Diagnosis Given'] == True]['EID'])
	subord = np.random.choice(subord, nsubs, replace=False)
	agel,pcl,phenol = make_phenol(subord)
	ISCfs = ISCpath+'ISC_small/ISC_small_'+subord[-1].split('/')[-1][4:-3]+'_'
	nshuff = 0 # number of shuffles
	
del phenol['all']
phenolperm = phenol
conds = ['age','sex'] #phenol.keys() (only analyzing age and sex!)

nsh = 1 #5 split-half iterations

def ISCe_calc(fstr):
	ISCe = dd.io.load(fstr,'/'+'ISC_w')[0]-dd.io.load(fstr,'/'+'ISC_w')[1]
	return ISCe
def ISCg_calc(fstr):
    ISCg = sum(dd.io.load(fstr,'/'+'ISC_b'))/4/(np.sqrt(dd.io.load(fstr,'/'+'ISC_w')[0])*
				   np.sqrt(dd.io.load(fstr,'/'+'ISC_w')[1]))
    return ISCg

def shuff_check(fstr,nshuff,good_v_indexes):
	fstrtmp = ('_').join(fstr.split('_')[:-1])+'_'+str(0)+'.h5'
	ISCg = [v for i,v in ISCg_calc(fstrtmp) if i in good_v_indexes]
	ISCe = [v for i,v in ISCe_calc(fstrtmp) if i in good_v_indexes]
	vvecte = np.zeros((nshuff,len(ISCe)))
	vvectg = np.zeros((nshuff,len(ISCg)))
	for shuff in np.arange(1,nshuff+1):
		fstrtmp = ('_').join(fstr.split('_')[:-1])+'_'+str(shuff)+'.h5'
		vvecte[shuff-1] = ISCe_calc(fstrtmp)
		vvectg[shuff-1] = ISCg_calc(fstrtmp)
	vertsg = np.asarray([np.sum(vvectg[:,v]<ISCg[v])/nshuff for v in range(len(ISCg))]) #(previously: np.sum(vvect<ISC[v]))
	vertse = np.asarray([np.sum(abs(vvecte[:,v])>abs(ISCe[v]))/nshuff for v in range(len(ISCe))])
	vertsidxe = [good_v_indexes[i] for i in [i for i, v in enumerate(vertse) \
				 if v<0.1 and ~np.isnan(ISCe[i]) and vvecte[-1,i]!=0]]
	vertsidxg = [good_v_indexes[i] for i in [i for i, v in enumerate(vertsg) \
				 if v<0.1 and ~np.isnan(ISCg[i]) and vvectg[-1,i]!=0]]
	good_v_indexes = list(set(vertsidxe+vertsidxg))
	return good_v_indexes


for s in range(nsh):
	phenol_split = {key:{key:[] for key in conds} for key in conds}
	subs = {key:[] for key in conds}
	for k in conds:
		if len(glob.glob(ISCfs+'*'+k+'*'))>0: # use same subjects as previous shuffs
			subs[k] = dd.io.load(glob.glob(ISCfs+'*'+k+'_shuff_0.h5')[0],'/subord')
			phenol_split[k] = dd.io.load(glob.glob(ISCfs+'*'+k+'_shuff_0.h5')[0],'/phenol')
		else: # create standardized subject list
			v2 = phenolperm['sex'] if k!='sex' else phenolperm['age']
			subh = even_out(phenolperm[k],v2)
			subs_idx = [item2 for sublist in subh for item in sublist for item2 in item] # keeping subjects CONSISTENT across shuffles!
			subs[k] = [sub for i,sub in enumerate(subord) if i in subs_idx]
			for cond2 in conds:
				phenol_split[k][cond2] = [p for i,p in enumerate(phenolperm[cond2]) if i in subs_idx]
	for task in ['DM','TP']:
		print(task)
		n_time = dd.io.load(subord[0],['/'+task+'/L'])[0].shape[1]
		n_vox = 81924
		for k in conds:
			fstr = ISCfs+task+'_'+k
			subord_k = subs[k]
			phenol = phenol_split[k]
			good_v_indexes = np.arange(n_vox)
			shuffl = np.arange(nshuff+1) # how many shuffs are left to run?
			if len(glob.glob(fstr+'*')) > 0:
				shuffcomp = [int(f.split('_')[-1][:-3]) for f in glob.glob(fstr+'*')]
				shuffl = np.asarray([s for s in shuffl if s not in shuffcomp])
				good_v_indexes = dd.io.load(fstr+'_shuff_'+str(np.max(shuffcomp))+'.h5',\
											'/good_v_indexes')
				# randomly shuffle phenol:
				for cond,v in phenol.items():
					nonnanidx = np.argwhere(~np.isnan(phenol[cond]))
					randidx = np.random.permutation(nonnanidx)
					phenol[cond] = [v[randidx[nonnanidx==idx][0]] if idx in nonnanidx else i for idx,i in enumerate(v)]
			for shuff in tqdm.tqdm(shuffl):
				fstr = ISCfs+task+'_'+k+'_shuff_'+str(shuff)+'.h5'
				shuffdict = {'subord':[],'subh':[],'phenol':[],'ISC_w':[],'ISC_b':[],
							'good_v_indexes':good_v_indexes}
				good_v_indexes = shuffdict['good_v_indexes']
				n_vox = len(good_v_indexes)
				v2 = phenol_split[k]['sex'] if k!='sex' else phenol_split[k]['age']
				shuffdict['phenol'] = phenol
				shuffdict['subord'] = subord_k
				shuffdict['subh'] = even_out(phenol[k],v2)
				groups = np.zeros((2,2,n_vox,n_time),dtype='float16')
				for h in [0,1]: # split all or between T / F
					for htmp in [0,1]: # split within split
						group = np.zeros((n_vox,n_time),dtype='float16')
						groupn = np.ones((n_vox,n_time),dtype='int')*len(shuffdict['subh'][h][htmp])
						for i in shuffdict['subh'][h][htmp]:
							D = np.concatenate([dd.io.load(subord_k[i],['/'+task+'/L'])[0], dd.io.load(subord_k[i],['/'+task+'/R'])[0]], axis=0)
							group = np.nansum(np.stack((group,\
														D[good_v_indexes,:])),axis=0)
							nanverts = np.argwhere(np.isnan(D[good_v_indexes,:]))
							groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
						groups[h,htmp] = zscore(group/groupn,axis=1)
					shuffdict['ISC_w'].append(np.sum(np.multiply(groups[h,0],groups[h,1]),axis=1)/(n_time-1))
				for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						shuffdict['ISC_b'].append(np.sum(np.multiply(groups[0,htmp1],groups[1,htmp2]),axis=1)/(n_time-1)) # correlate across groups
				if any(shu==shuff for shu in [101,1001]):
					shuffdict['good_v_indexes'] = shuff_check(fstr,shuff-1,good_v_indexes)
					print('The number of verts left after',str(shuff),\
						  'iterations for group',k,'is',len(good_v_indexes))
				dd.io.save(fstr,shuffdict)	
				# randomly shuffle phenol:
				for cond,v in phenol.items():
					nonnanidx = np.argwhere(~np.isnan(phenol[cond]))
					randidx = np.random.permutation(nonnanidx)
					phenol[cond] = [v[randidx[nonnanidx==idx][0]] if idx in nonnanidx else i for idx,i in enumerate(v)]
		
		