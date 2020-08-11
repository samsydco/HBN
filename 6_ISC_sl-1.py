#!/usr/bin/env python3

# ISC Calculation #1:
# Calculate ISC per-voxel in each ROI/searchlight and then average

import os
import glob
import tqdm
import random
import numpy as np
import deepdish as dd
from scipy.stats import zscore
from ISC_settings import *
SLlist = dd.io.load(ISCpath+'SLlist.h5')

nTR=[750,250]
bins = [0,4]
nvox = 81924//2
savedir = ISCpath+'SL_1/'
nsub = 41

def ISCe_calc(etcdict):
	ISCe = etcdict['ISC_w'][1]-etcdict['ISC_w'][0]
	return ISCe
def ISCg_calc(etcdict):
    ISCg = sum(etcdict['ISC_b'])/4/(np.sqrt(etcdict['ISC_w'][1])*
				   np.sqrt(etcdict['ISC_w'][0]))
    return ISCg

for ti,task in enumerate(['DM','TP']):
	n_time = nTR[ti]
	for hem in ['L','R']:
		print(task,hem)
		subsavedir = savedir+task+'/'+hem+'/'
		SLs = SLlist[hem]
		SLs = {key: SLs[key] for key in np.arange(len(SLs))}
		etcdict = {key: {} for key in np.arange(len(SLs))}
		SLdict  = {key: {key:[] for key in np.arange(nshuff+1)} \
				   for key in ['e_diff','g_diff']}
		voxdict = {key: np.zeros((nshuff+1,nvox)) for key in ['e_diff','g_diff']}
		voxcount = np.zeros(nvox)
		if not os.path.exists(subsavedir):
			os.makedirs(subsavedir)
		elif len(glob.glob(subsavedir+'*')) > 0:
			maxSL = np.max([int(SL.split('_')[-1][:-3]) for SL in glob.glob(subsavedir+'*')])
			SLs = dict((k,SLs[k]) for k in SLs.keys() if k > maxSL)
			loaddict = dd.io.load(subsavedir+'_'.join([task,hem,str(maxSL)])+'.h5')
			etcdict = loaddict['etcdict']
			SLdict  = loaddict['SLdict']
			voxdict = loaddict['voxdict']
			voxcount = loaddict['voxcount']
		D = np.empty((nsub*2,nvox,n_time),dtype='float16')
		Age = []
		Sex = []
		for bi,b in enumerate(bins):
			subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
			Sex.extend([Phenodf['Sex'][Phenodf['EID'] == shortsub(sub)].iloc[0] for sub in subl])
			Age.extend([bi]*len(subl))
			sub_ = 0 if b==0 else nsub # young and then old in D
			for sidx, sub in enumerate(subl):
				D[sidx+sub_] = dd.io.load(sub,['/'+task+'/'+hem])[0]
		Ageperm = Age
		for vi,voxl in tqdm.tqdm(SLs.items()):
			n_vox = len(voxl)
			etcdict[vi]['voxl'] = voxl
			voxcount[voxl] += 1
			Dsl = D[:,voxl,:]
			Age = Ageperm
			for shuff in range(nshuff+1):
				shuffstr = 'shuff_'+str(shuff)
				etcdict[vi][shuffstr] = {'ISC_w':[],'ISC_b':[]}
				subh = even_out(Age,Sex)
				groups = np.zeros((2,2,n_vox,n_time),dtype='float16')
				for h in [0,1]:
					for htmp in [0,1]:
						group = np.zeros((n_vox,n_time),dtype='float16')
						groupn = np.ones((n_vox,n_time),dtype='int')*nsub//2
						for i in subh[h][htmp]:
							group = np.nansum(np.stack((group,Dsl[i])),axis=0)
							nanverts = np.argwhere(np.isnan(Dsl[i,:]))
							groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
						groups[h,htmp] = zscore(group/groupn,axis=1)
					etcdict[vi][shuffstr]['ISC_w'].append(np.mean(np.sum(np.multiply(groups[h,0],groups[h,1]),axis=1)/(n_time-1)))
				for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						etcdict[vi][shuffstr]['ISC_b'].append(np.mean(np.sum(np.multiply(groups[0,htmp1],groups[1,htmp2]),axis=1)/(n_time-1)))
				# Now calculate g_diff and e_diff
				e_diff = np.nanmean(ISCe_calc(etcdict[vi][shuffstr]))
				g_diff = np.nanmean(ISCg_calc(etcdict[vi][shuffstr]))
				SLdict['e_diff'][shuff].append(e_diff)
				voxdict['e_diff'][shuff,voxl] += e_diff
				SLdict['g_diff'][shuff].append(g_diff)
				voxdict['g_diff'][shuff,voxl] += g_diff
				# Now shuffle Age:
				random.shuffle(Age)
			dd.io.save(subsavedir+'_'.join([task,hem,str(vi)])+'.h5',\
					   {'voxdict':voxdict, 'SLdict':SLdict, 'voxcount':voxcount,'etcdict':etcdict})
		for k,v in voxdict.items():
			voxdict[k] = v / voxcount
		dd.io.save(savedir+'_'.join([task,hem])+'.h5', \
				   {'voxdict':voxdict, 'SLdict':SLdict, 'voxcount':voxcount, 'etcdict':etcdict})
			
				





