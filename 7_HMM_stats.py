#!/usr/bin/env python3

# Use ll values determined in seperate code
# Permute subject IDs between young and old
# At k value with best ll compute:
# (a) ll difference between "young" and "old"
# (b) auc difference between "young" and "old"
# determine how fast 100 shuffs takes to determine if can run more,
# or need to do normal distribution approximation...

'''
New Process:

1) Get patterns by HMM([Young_train, Old_train])
2) LLs = find_events(Young_test), find_events(Old_test)
3) Pick K by maximizing avg LLs
4A) LL diff at this K
4B) AUC diff at this K
Permutations only for 4
Perm: find_events(Perm1), find_events(Perm2)
All of this train/test set loop

'''

import glob
import tqdm
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
from sklearn.model_selection import KFold
from HMM_settings import *

ROInow = ROIopts[1]
HMMf = HMMpath+'timing_'+ROInow+'/'
savedir = HMMpath+'shuff/'
ROIs = glob.glob(HMMf+'*h5')
nsub = 41
kf = KFold(n_splits=nsplit,shuffle=True)

for roi in tqdm.tqdm(ROIs):
	roi_short = roi.split('/')[-1][:-3]
	ROIsHMM = dd.io.load(roi)
	roidict = {key: {} for key in tasks}
	for ti,task in enumerate(tasks):
		nTR_ = nTR[ti]
		#Now calculating best_k from average ll:
		best_k = k_list[np.argmax(np.mean(np.concatenate((ROIsHMM[task]['bin_0']['tune_ll'],ROIsHMM[task]['bin_4']['tune_ll'])),0))]
		D = [ROIsHMM[task]['bin_0']['D'],ROIsHMM[task]['bin_4']['D']]
		tune_seg = np.zeros((nshuff,2,nsplit,nTR_,best_k))
		tune_ll = np.zeros((nshuff,2,nsplit))
		for split in range(nsplit):
			LI,LO = next(kf.split(np.arange(nsub)))
			Dtrain = [np.mean(D[0][LI],axis=0).T,
					  np.mean(D[1][LI],axis=0).T]
			Dtest_all =  np.concatenate([D[0][LO],D[1][LO]])
			nsubLO = len(LO)
			subl = np.arange(nsubLO*2) # subject list to be permuted!
			hmm = brainiak.eventseg.event.EventSegment(n_events=best_k)
			hmm.fit(Dtrain)
			for shuff in range(nshuff+1):
				Dtest = [np.mean(Dtest_all[subl[:nsubLO]],axis=0).T,
						 np.mean(Dtest_all[subl[nsubLO:]],axis=0).T]
				for bi in range(len(bins)):
					tune_seg[shuff,bi,split], tune_ll[shuff,bi,split] = hmm.find_events(Dtest[bi])
				# RANDOMIZE
				subl = np.random.permutation(nsubLO*2)
		for shuff in range(nshuff+1):
			shuffstr = 'shuff_'+str(shuff)
			roidict[task][shuffstr] = {}		
			auc = []
			for bi in range(len(bins)):
				auc.append(np.dot(np.mean(tune_seg[shuff,bi],axis=0), np.arange(best_k)).sum())
			roidict[task][shuffstr]['auc_diff'] = (auc[1]-auc[0])/(best_k)*TR
			roidict[task][shuffstr]['ll_diff'] = np.mean(tune_ll[shuff,1,:]) - np.mean(tune_ll[shuff,0,:])
	dd.io.save(savedir+roi_short+'.h5',roidict)
		

	
		
		
	
	