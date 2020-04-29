#!/usr/bin/env python3

import os
import glob
import tqdm
import random
import numpy as np
import deepdish as dd
from scipy.stats import pearsonr
from ISC_settings import *
SLlist = dd.io.load(ISCpath+'SLlist.h5')

nTR=[750,250]
bins = [0,4]
nvox = 81924//2
savedir = ISCpath+'SL/'
nsub = 41

def ISCe_calc(etcdict):
	ISCe = etcdict['ISC_w'][1]-etcdict['ISC_w'][0]
	return ISCe
def ISCg_calc(etcdict):
    ISCg = sum(etcdict['ISC_b'])/4/(np.sqrt(etcdict['ISC_w'][1])*
				   np.sqrt(etcdict['ISC_w'][0]))
    return ISCg

for ti,task in enumerate(['DM','TP']):
	nTR_ = nTR[ti]
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
		D = np.empty((nsub*2,nvox,nTR_),dtype='float16')
		Age = []
		Sex = []
		for bi,b in enumerate(bins):
			subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
			Sex.extend([Phenodf['Sex'][Phenodf['EID'] == shortsub(sub)].iloc[0] for sub in subl])
			Age.extend([bi]*len(subl))
			sub_ = 0 if b==0 else nsub # young and then old in D
			for sidx, sub in enumerate(subl):
				D[sidx+sub_] = dd.io.load(sub,['/'+task+'/'+hem])[0]
		for vi,voxl in tqdm.tqdm(SLs.items()):
			badvox = np.unique(np.where(np.isnan(D[:,voxl,:]))[1])
			voxl_tmp = np.array([v for i,v in enumerate(voxl) if not any(b==i for b in badvox)])
			etcdict[vi]['voxl'] = voxl_tmp
			voxcount[voxl_tmp] += 1
			Dsl = D[:,voxl_tmp,:]
			
			for shuff in range(nshuff+1):
				shuffstr = 'shuff_'+str(shuff)
				etcdict[shuffstr] = {'ISC_w':[],'ISC_b':[]}
				subh = even_out(Age,Sex)
				groups = np.zeros((2,2,nTR_),dtype='float16')
				for h in [0,1]:
					for htmp in [0,1]:
						for i in subh[h][htmp]:
							groups[h,htmp] = np.nansum(np.stack((groups[h,htmp],np.mean(Dsl[i],0))),axis=0)
					etcdict[shuffstr]['ISC_w'].append(pearsonr(groups[h,0],groups[h,1])[0])
				for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						etcdict[shuffstr]['ISC_b'].append(pearsonr(groups[0,htmp1],groups[1,htmp2])[0])
				# Now calculate g_diff and e_diff
				e_diff = ISCe_calc(etcdict[shuffstr])
				g_diff = ISCg_calc(etcdict[shuffstr])
				SLdict['e_diff'][shuff].append(e_diff)
				voxdict['e_diff'] += e_diff
				SLdict['g_diff'][shuff].append(g_diff)
				voxdict['g_diff'] += g_diff
				# Now shuffle Age:
				random.shuffle(Age)
			dd.io.save(subsavedir+'_'.join([task,hem,str(vi)])+'.h5',\
					   {'voxdict':voxdict, 'SLdict':SLdict, 'voxcount':voxcount,'etcdict':etcdict})
		for k,v in voxdict.items():
			voxdict[k] = v / voxcount
		dd.io.save(savedir+'_'.join([task,hem])+'.h5', \
				   {'voxdict':voxdict, 'SLdict':SLdict, 'voxcount':voxcount, 'etcdict':etcdict})
			
				





