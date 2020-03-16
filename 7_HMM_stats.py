#!/usr/bin/env python3

# Use ll values determined in seperate code
# Permute subject IDs between young and old
# At k value with best ll compute:
# (a) ll difference between "young" and "old"
# (b) auc difference between "young" and "old"
# determine how fast 100 shuffs takes to determine if can run more,
# or need to do normal distribution approximation...


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
	for task in tasks:
		best_k = k_list[np.argmax(np.mean(ROIsHMM[task]['tune_ll'],axis=0))]
		D = np.concatenate([ROIsHMM[task]['bin_0']['D'],ROIsHMM[task]['bin_4']['D']])
		subl = np.arange(nsub*2)
		for shuff in tqdm.tqdm(range(nshuff+1)):
			shuffstr = 'shuff_'+str(shuff)
			roidict[task][shuffstr] = {}
			Dsplit = [D[subl[:nsub]],D[subl[nsub:]]] # split young and old
			hmm = brainiak.eventseg.event.EventSegment(n_events=best_k)
			hmm.fit([np.mean(d,axis=0).T for d in Dsplit])
			auc = []
			for bi in range(len(bins)):
				auc.append(np.dot(hmm.segments_[bi], np.arange(best_k)).sum())
			roidict[task][shuffstr]['auc_diff'] = (auc[1]-auc[0])/(best_k)*TR
			for b in bins:
				roidict[task][shuffstr]['bin_'+str(b)] = np.zeros(nsplit)
			for split in range(nsplit):
				LI,LO = next(kf.split(np.arange(nsub)))
				Dtrain = [np.mean(Dsplit[0][LI],axis=0).T,
						  np.mean(Dsplit[1][LI],axis=0).T]
				Dtest =  [np.mean(Dsplit[0][LO],axis=0).T,
						  np.mean(Dsplit[1][LO],axis=0).T]
				hmm = brainiak.eventseg.event.EventSegment(n_events=best_k)
				hmm.fit(Dtrain)
				for bi,b in enumerate(bins):
					_, tune_ll = hmm.find_events(Dtest[bi]) # tune_ll per age group
					roidict[task][shuffstr]['bin_'+str(b)][split] = tune_ll
			# RANDOMIZE
			subl = np.random.permutation(nsub*2)
	dd.io.save(savedir+roi_short+'.h5',roidict)
		

	
		
		
	
	