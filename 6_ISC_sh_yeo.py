#!/usr/bin/env python3

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
nsub = 41

for roi in tqdm.tqdm(glob.glob(roidir+'*.h5')):
	roi_short = roi.split('/')[-1][:-3]
	roidict = {}
	for ti,task in enumerate(['DM']):
		vall = dd.io.load(roi,'/vall')
		n_vox = len(vall)
		n_time = nTR[ti]
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
		roidict[task] = {'vall':vall, 'ISC_w':np.zeros((nshuff+1,nbins,n_vox)), 'ISC_w_time':np.zeros((nshuff+1,nbins,n_vox,n_time)), 'ISC_g_time':np.zeros((nshuff+1,n_vox,n_time)), 'ISC_g':np.zeros((nshuff+1,n_vox)), 'ISC_e':np.zeros((nshuff+1,n_vox))}
		for shuff in range(nshuff+1):
			subh = even_out(Age,Sex)
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
				roidict[task]['ISC_w_time'][shuff,h] = np.multiply(groups[h,0],groups[h,1])
				roidict[task]['ISC_w'][shuff,h] = np.sum(roidict[task]['ISC_w_time'][shuff,h],axis=1)/(n_time-1)
			ISC_b_time = []
			for htmp1 in [0,1]:
				for htmp2 in [0,1]:
					ISC_b_time.append(np.multiply(groups[0,htmp1], groups[1,htmp2]))
			# Now calculate g_diff and e_diff
			ISCg_time = np.sum(ISC_b_time,axis=0)
			denom = np.sqrt(roidict[task]['ISC_w'][shuff,0]) * \
					np.sqrt(roidict[task]['ISC_w'][shuff,1])
			roidict[task]['ISC_g_time'][shuff] = np.sum(ISC_b_time, axis=0)/4/np.tile(denom,(n_time,1)).T
			roidict[task]['ISC_g'][shuff] = np.sum([np.sum(i,axis=1)/(n_time-1) for i in ISC_b_time], axis=0)/4/denom
			roidict[task]['ISC_e'][shuff] = roidict[task]['ISC_w'][shuff,0] - \
					roidict[task]['ISC_w'][shuff,1]
			# Now shuffle Age, and Sex in same order:
			neword = np.random.permutation(len(Age))
			Age = [Age[neword[ai]] for ai,a in enumerate(Age)]
			Sex = [Sex[neword[ai]] for ai,a in enumerate(Sex)]
	dd.io.save(savedir+roi_short+'.h5',roidict)
	
	
# Do any ROIs significantly correlate with ev_conv?
from HMM_settings import *
sigroig = {}
sigroie = {}
for roi in tqdm.tqdm(glob.glob(savedir+'*.h5')):
	roi_short = roi.split('/')[-1][:-3]
	ISC_g_time = dd.io.load(roi,'/DM/ISC_g_time')
	ISC_e_time = dd.io.load(roi,'/DM/ISC_w_time')[:,1]-dd.io.load(roi,'/DM/ISC_w_time')[:,0]
	nshuff = ISC_g_time.shape[0]
	rlg = np.zeros(nshuff)
	rle = np.zeros(nshuff)
	for shuff in range(nshuff):
		rlg[shuff],p = pearsonr(np.nanmean(ISC_g_time[shuff],axis=0),ev_conv)
		rle[shuff],p = pearsonr(np.nanmean(ISC_e_time[shuff],axis=0),ev_conv)
	pvalg = np.sum(rlg[0]>rlg[1:])/(nshuff-1)
	pvale = np.sum(rle[0]>rle[1:])/(nshuff-1)
	if pvalg < 0.05:
		sigroig[roi_short] = {'r':rlg[0],'p':pvalg}
	if pvale < 0.05:
		sigroie[roi_short] = {'r':rle[0],'p':pvale}
		

				
				
			
				
			
			
			
			
			