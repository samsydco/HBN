#!/usr/bin/env python3

import os
import glob
import tqdm
import numpy as np
import deepdish as dd
from scipy.stats import zscore, pearsonr
from ISC_settings import *

nTR=[750,250]
bins = [0,4]
nbins = len(bins)
roidir = ISCpath+'Yeo_parcellation/'
savedir = ISCpath+'shuff_Yeo/'
nsub = 40
nshuff2perm=1000

def p_calc(ISC,ISCtype='e'):
	nshuff = ISC.shape[0]-1
	if ISCtype == 'e':
		p = np.sum(abs(np.nanmean(ISC[0]))<abs(np.nanmean(ISC[1:],axis=1)))/nshuff
	else:
		p = np.sum(np.nanmean(ISC[0])>np.nanmean(ISC[1:],axis=1))/nshuff
	return p,nshuff

def load_D(roi,task,bins):
	D = []
	Age = []
	Sex = []
	for bi,b in enumerate(bins):
		bstr = 'bin_'+str(b)
		subl = dd.io.load(roi,'/'+'/'.join([task,bstr,'subl']))
		Sex.extend([Phenodf['Sex'][Phenodf['EID'] == shortsub(sub)].iloc[0] for sub in subl])
		Age.extend([bi]*len(subl))
		D.append(dd.io.load(roi,'/'+'/'.join([task,bstr,'D'])))
	D = np.concatenate(D)
	return D,Age,Sex

def shuff_demo(Age,Sex):
	# Now shuffle Age, and Sex in same order:
	neword = np.random.permutation(len(Age))
	Age = [Age[neword[ai]] for ai,a in enumerate(Age)]
	Sex = [Sex[neword[ai]] for ai,a in enumerate(Sex)]
	return Age,Sex
	
def ISC_w_calc(D,n_vox,n_time,nsub,subh):
	ISC_w = np.zeros((nbins,n_vox))
	groups = np.zeros((nbins,2,n_vox,n_time),dtype='float16')
	for h in [0,1]:
		for htmp in [0,1]:
			group = np.zeros((n_vox,n_time),dtype='float16')
			groupn = np.ones((n_vox,n_time),dtype='int')*nsub//2
			for i in subh[h][htmp]:
				group = np.nansum(np.stack((group,D[i])),axis=0)
				nanverts = np.argwhere(np.isnan(D[i,:]))
				groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
			groups[h,htmp] = zscore(group/groupn,axis=1)
		ISC_w[h] = np.sum(np.multiply(groups[h,0],groups[h,1]), axis=1)/(n_time-1)
	return ISC_w,groups

for roi in tqdm.tqdm(glob.glob(roidir+'*.h5')):
	roi_short = roi.split('/')[-1][:-3]
	if os.path.exists(savedir+roi_short+'.h5'):
		roidict = dd.io.load(savedir+roi_short+'.h5')
	else:
		roidict = {}
	for ti,task in enumerate(['DM']):
		vall = dd.io.load(roi,'/vall')
		n_vox = len(vall)
		n_time = nTR[ti]
		D,Age,Sex = load_D(roi,task,bins)
		shuffl = 0
		if os.path.exists(savedir+roi_short+'.h5'):
			taskv = roidict[task]
			e_p,nshuff_ = p_calc(taskv['ISC_e'],'e')
			g_p,nshuff_ = p_calc(taskv['ISC_g'],'g')
			nshuff2 = nshuff2perm + nshuff_
			if e_p < 0.05 or g_p < 0.05:
				roidict[task]['ISC_w'] = np.append(taskv['ISC_w'], np.zeros((nshuff2-nshuff_,nbins,n_vox)), axis=0)
				roidict[task]['ISC_e'] = np.append(taskv['ISC_e'], np.zeros((nshuff2-nshuff_,n_vox)), axis=0)
				roidict[task]['ISC_b'] = np.append(taskv['ISC_b'], np.zeros((nshuff2-nshuff_,4,n_vox)), axis=0)
				roidict[task]['ISC_g'] = np.append(taskv['ISC_g'], np.zeros((nshuff2-nshuff_,n_vox)), axis=0)
				roidict[task]['ISC_g_time'] = np.append(taskv['ISC_g_time'], np.zeros((nshuff2-nshuff_,n_vox,n_time)), axis=0)
				shuffl = np.arange(nshuff_+1,nshuff2+1)
		else:
			roidict[task] = {'vall':vall, 'ISC_w':np.zeros((nshuff+1,nbins,n_vox)), 'ISC_b':np.zeros((nshuff+1,4,n_vox)), 'ISC_g_time':np.zeros((nshuff+1,n_vox,n_time)), 'ISC_g':np.zeros((nshuff+1,n_vox)), 'ISC_e':np.zeros((nshuff+1,n_vox))}
			shuffl = np.arange(nshuff+1)
		for shuff in shuffl:
			if shuff !=0:
				Age,Sex = shuff_demo(Age,Sex)
			subh = even_out(Age,Sex)
			roidict[task]['ISC_w'][shuff], groups =\
				ISC_w_calc(D,n_vox,n_time,nsub,subh)
			roidict[task]['ISC_e'][shuff] = roidict[task]['ISC_w'][shuff,0] -\
											roidict[task]['ISC_w'][shuff,1]
			ISC_b_time = []
			idx = 0
			for htmp1 in [0,1]:
				for htmp2 in [0,1]:
					ISC_b_time.append(np.multiply(groups[0,htmp1], groups[1,htmp2]))
					roidict[task]['ISC_b'][shuff,idx] = [np.sum(ISC_b_time[-1], axis=1)/(n_time-1)
					idx+=1
			denom = np.sqrt(roidict[task]['ISC_w'][shuff,0]) * \
					np.sqrt(roidict[task]['ISC_w'][shuff,1])
			roidict[task]['ISC_g_time'][shuff] = np.sum(ISC_b_time, axis=0)/4/np.tile(denom,(n_time,1)).T
			roidict[task]['ISC_g'][shuff] = np.sum(roidict[task]['ISC_b'][shuff], axis=0)/4/denom
	dd.io.save(savedir+roi_short+'.h5',roidict)

														 
														 