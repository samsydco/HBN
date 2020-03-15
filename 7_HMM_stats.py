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
from HMM_settings import *

ROInow = ROIopts[1]
HMMf = HMMpath+'timing_'+ROInow+'/'
ROIs = glob.glob(HMMf+'*h5')

for roi in tqdm.tqdm(ROIs):
	roi_short = roi.split('/')[-1][:-3]
	ROIsHMM = dd.io.load(roi)
	for task in tasks:
		best_k = k_list[np.argmax(np.mean(ROIsHMM[task]['tune_ll'],axis=0))]
		
		for b in bins:
			ROIsHMM[task]['bin_'+str(b)]['tune_ll'] = np.zeros((nsplit,len(k_list)))
		for split in range(nsplit):
			splitsrt = 'split_'+str(split)
			LI,LO = next(kf.split(np.arange(nsub)))
			Dtrain = [np.mean(ROIsHMM[task]['bin_0']['D'][LI],axis=0).T,
					  np.mean(ROIsHMM[task]['bin_4']['D'][LI],axis=0).T]
			Dtest =  [np.mean(ROIsHMM[task]['bin_0']['D'][LO],axis=0).T,
					  np.mean(ROIsHMM[task]['bin_4']['D'][LO],axis=0).T]
			# Fit HMM with TxV data, leaving some subjects out
			for ki,k in enumerate(k_list):
				kstr = 'k_'+str(k)
				hmm = brainiak.eventseg.event.EventSegment(n_events=k)
				hmm.fit(Dtrain)
				ROIsHMM[task][splitsrt][kstr]['train_ll']=hmm.ll_
				# predict the event boundaries for each bin
				for bi,b in enumerate(bins):
					_, tune_ll = hmm.find_events(Dtest[bi])
					ROIsHMM[task]['bin_'+str(b)]['tune_ll'][split,ki] = tune_ll
	
		
		
	
	