#!/usr/bin/env python3

import glob
import tqdm
import random
import numpy as np
import deepdish as dd
from scipy.stats import zscore
import nibabel.freesurfer.io as free
from ISC_settings import *

nTR=[750,250]
bins = [0,4]
HMMdir = HMMpath+'shuff/'
savedir = ISCpath+'shuff_Yeo/'
nsub = 41

def ISCe_calc(roidict,task,shuffstr):
	ISCe = roidict[task][shuffstr]['ISC_w'][1]-roidict[task][shuffstr]['ISC_w'][0]
	return ISCe
def ISCg_calc(roidict,task,shuffstr):
    ISCg = sum(roidict[task][shuffstr]['ISC_b'])/4/(np.sqrt(roidict[task][shuffstr]['ISC_w'][1])*
				   np.sqrt(roidict[task][shuffstr]['ISC_w'][0]))
    return ISCg

for roi in tqdm.tqdm(glob.glob(HMMdir+'*.h5')):
	roi_short = roi.split('/')[-1][:-3]
	h = roi_short[0]
	vall = dd.io.load(roi,'/vall')
	n_vox = len(vall)
	roidict = {}
	roidict['vall'] = vall
	for ti,task in enumerate(tasks):
		roidict[task] = {}
		n_time = nTR[ti]
		D = []
		Age = []
		Sex = []
		for bi,b in enumerate(bins):
			bstr = 'bin_'+str(b)
			subl = dd.io.load(HMMdir+roi_short+'.h5','/'+'/'.join([task,bstr,'subl']))
			Sex.extend([Phenodf['Sex'][Phenodf['EID'] == shortsub(sub)].iloc[0] for sub in subl])
			Age.extend([bi]*len(subl))
			D.append(dd.io.load(HMMdir+roi_short+'.h5','/'+'/'.join([task,bstr,'D'])))
		D = np.concatenate(D)
		for shuff in range(nshuff+1):
			shuffstr = 'shuff_'+str(shuff)
			roidict[task][shuffstr] = {'ISC_w':[],'ISC_b':[]}
			subh = even_out(Age,Sex)
			groups = np.zeros((2,2,n_vox,n_time),dtype='float16')
			for h in [0,1]:
				for htmp in [0,1]:
					group = np.zeros((n_vox,n_time),dtype='float16')
					groupn = np.ones((n_vox,n_time),dtype='int')*nsub//2
					for i in subh[h][htmp]:
						group = np.nansum(np.stack((group,D[i])),axis=0)
						nanverts = np.argwhere(np.isnan(D[i,:]))
						groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
					groups[h,htmp] = zscore(group/groupn,axis=1)
				roidict[task][shuffstr]['ISC_w'].append(np.sum(np.multiply(groups[h,0],groups[h,1]),axis=1)/(n_time-1))
			for htmp1 in [0,1]:
				for htmp2 in [0,1]:
					roidict[task][shuffstr]['ISC_b'].append(np.sum(np.multiply(groups[0,htmp1],groups[1,htmp2]),axis=1)/(n_time-1))
			# Now calculate g_diff and e_diff
			roidict[task][shuffstr]['e_diff'] = ISCe_calc(roidict,task,shuffstr)
			roidict[task][shuffstr]['g_diff'] = ISCg_calc(roidict,task,shuffstr)
			# Now shuffle Age:
			random.shuffle(Age)
	dd.io.save(savedir+roi_short+'.h5',roidict)
				
				
				
			
				
			
			
			
			
			