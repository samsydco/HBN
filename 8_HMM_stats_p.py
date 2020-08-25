#!/usr/bin/env python3

# Add more shuffles for ROIs with low p-values:

import os
import time
import glob
import tqdm
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
from sklearn.model_selection import KFold
from HMM_settings import *

newsavedir = HMMpath+'shuff_5bins_train04/'
nsub=41
y = [0]*int(np.floor(nsub/nsplit))*4+[1]*(int(np.floor(nsub/nsplit))+1)
kf = KFold(n_splits=nsplit, shuffle=True, random_state=2)
bins = np.arange(nbinseq)
nbins = len(bins)
nshuff2=1000


for roi in tqdm.tqdm(glob.glob(newsavedir+'*.h5')):
	#if 'May' in time.ctime(os.path.getmtime(roi)):
	roidict = dd.io.load(roi)
	for task,taskv in roidict.items():
		nshuff = len([k for k in list(taskv.keys()) if 'shuff' in k]) - 1
		p_ll_ = np.sum(abs(taskv['shuff_0']['ll_diff'])<[abs(taskv['shuff_'+str(s)]['ll_diff']) for s in range(1,nshuff+1)])/nshuff
		p_auc = np.sum(abs(taskv['shuff_0']['auc_diff'])<[abs(taskv['shuff_'+str(s)]['auc_diff']) for s in range(1,nshuff+1)])/nshuff
		if p_ll_<0.05 or p_auc<0.05:
			print(roi,task)
			bin_tmp = bins if 'all' in newsavedir else [0,4]
			D = [taskv['bin_'+str(b)]['D'] for b in bins]
			nsub,_,nTR_ = D[0].shape
			best_k = taskv['best_k']
			tune_seg = np.append(taskv['tune_seg_perm'],np.zeros((nshuff2-nshuff,nbins,nsplit,nTR_,best_k)),axis=0)
			tune_ll = np.append(taskv['tune_ll_perm'],np.zeros((nshuff2-nshuff,nbins,nsplit)),axis=0)
			for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
				Dtrain = [np.mean(d[Ls[0]],axis=0).T for d in [D[bi] for bi in bin_tmp]]
				Dtest_all  = np.concatenate([d[Ls[1]] for d in D])
				nsubLO = len(Ls[1])
				subl = np.arange(nsubLO*nbins) # subject list to be permuted!
				hmm = brainiak.eventseg.event.EventSegment(n_events=best_k)
				hmm.fit(Dtrain)
				for shuff in np.arange(nshuff+1,nshuff2+1):
					# RANDOMIZE
					subl = np.random.permutation(nsubLO*nbins)
					for bi in range(nbins):
						idx = np.arange(bi*nsubLO,(bi+1)*nsubLO)
						Dtest = np.mean(Dtest_all[subl[idx]],axis=0).T
						tune_seg[shuff,bi,split], tune_ll[shuff,bi,split] = hmm.find_events(Dtest)		
			roidict[task]['tune_ll_perm'] = tune_ll
			roidict[task]['tune_seg_perm'] = tune_seg
			roidict[task]['E_k'] = np.zeros((nshuff2+1,nbins,nTR_))
			roidict[task]['auc'] = np.zeros((nshuff2+1,nbins))
			roidict[task]['auc_diff'] = np.zeros(nshuff2+1)
			roidict[task]['ll_diff'] = np.zeros(nshuff2+1)
			for shuff in range(nshuff2+1):
				for bi in range(nbins):
					roidict[task]['E_k'][shuff,bi] = np.dot(np.mean(tune_seg[shuff,bi],axis=0), np.arange(best_k)+1)
					roidict[task]['auc'][shuff,bi] = roidict[task]['E_k'][shuff,bi].sum()
				roidict[task]['auc_diff'][shuff] = ((roidict[task]['auc'][shuff,-1] -roidict[task]['auc'][shuff,0])/best_k)*TR
				roidict[task]['ll_diff'][shuff] = np.mean(np.diff(np.mean(tune_ll[shuff],axis=1)))/nTR_
		dd.io.save(roi,roidict)
