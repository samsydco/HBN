#!/usr/bin/env python3

# 1) compute a joint HMM fit over youngest and oldest datasets (1st average activity within group)
# 2) Get probability of being in each event at each time point for young group and old group (seperately)
# 3) plot the expected event (k) over time - determine if one age group lags behind the other

import os
import glob
import tqdm
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
from HMM_settings import *
from scipy.stats import pearsonr
from random import randrange
from sklearn.model_selection import KFold

ROIopts = ['YeoROIsforSRM_sel_2020-01-14.h5','YeoROIsforSRM_2020-01-03.h5','SfN_2019/ROIs_Fig3/Fig3_','g_diff/']
ROInow = ROIopts[0]
ROIfold = path+'ROIs/'+ROInow
HMMf = HMMpath+'timing_'+ROInow+'/'
if not os.path.exists(HMMf):
    os.makedirs(HMMf)
	
kf = KFold(n_splits=nsplit,shuffle=True)

ROIs = makeROIdict(ROIfold)
				
for roi in tqdm.tqdm(ROIs):
	ROIf = HMMf+roi+'.h5'
	if not os.path.exists(ROIf):
		ROIsHMM = {}
		vall = ROIs[roi]['vall']
		hemi = ROIs[roi]['hemi']
		ROIsHMM['hemi'] = hemi
		subl = [ageeq[i][1][b][idx] for b in bins for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
		nsub = len(subl)
		for task in ROIs[roi]['tasks']:
			ROIsHMM[task] = {}
			# Load data
			dtmp = dd.io.load(subl[0],['/'+task+'/'+hemi],sel=dd.aslice[vall,:])[0]
			vall = [v for i,v in enumerate(vall) if i not in np.where(np.isnan(dtmp[:,0]))[0]]
			ROIsHMM['vall'] = vall
			ROIsHMM['nvox'] = len(vall)
			nTR = dtmp.shape[1]
			D = np.empty((nsub,ROIsHMM['nvox'],nTR),dtype='float16')
			for sidx, sub in enumerate(subl):
				D[sidx,:,:] = dd.io.load(sub,['/'+task+'/'+hemi],sel=dd.aslice[vall,:])[0]
			# saveing measures of HMM fit for finding number of events
			ROIsHMM[task]['tune_ll'] = np.zeros((nsplit,len(k_list)))
			ROIsHMM[task]['within_r'] = np.zeros((nsplit,len(k_list),len(win_range)))
			ROIsHMM[task]['across_r'] = np.zeros((nsplit,len(k_list),len(win_range)))
			for split in range(nsplit):
				splitsrt = 'split_'+str(split)
				ROIsHMM[task][splitsrt] = {}
				LI,LO = next(kf.split(np.arange(nsub)))
				Dtrain = D[LI]
				Dtest = D[LO]
				# Fit HMM with TxV data, leaving some subjects out
				for ki,k in enumerate(k_list):
					kstr = 'k_'+str(k)
					ROIsHMM[task][splitsrt][kstr] = {}
					hmm = brainiak.eventseg.event.EventSegment(n_events=k)
					hmm.fit(np.mean(Dtrain,axis=0).T)
					ROIsHMM[task][splitsrt][kstr]['pattern']=hmm.event_pat_
					ROIsHMM[task][splitsrt][kstr]['seg_og']=hmm.segments_[0]
					ROIsHMM[task][splitsrt][kstr]['event_var']=hmm.event_var_
					# predict the event boundaries for the test set
					hmm_bounds, tune_ll = hmm.find_events(np.mean(Dtest, axis=0).T)
					ROIsHMM[task][splitsrt][kstr]['seg_lo']=hmm_bounds
					ROIsHMM[task]['tune_ll'][split,ki]=tune_ll[0]
					events = np.argmax(hmm_bounds, axis=1)
					_, event_lengths = np.unique(events, return_counts=True)
					hmm_bounds = np.where(np.diff(events))[0]
					# window size for within vs across correlations
					for wi,w in enumerate(win_range): # windows in range 5 - 10 sec
						corrs = np.zeros(nTR-w)
						for t in range(nTR-w):
							corrs[t] = pearsonr(np.mean(Dtest, axis=0)[:,t],\
												np.mean(Dtest, axis=0)[:,t+w])[0]
						# Test within minus across boudary pattern correlation with held-out subjects
						ROIsHMM[task]['within_r'][split,ki,wi] = np.mean(corrs[events[:-w] == events[w:]])
						ROIsHMM[task]['across_r'][split,ki,wi] = np.mean(corrs[events[:-w] != events[w:]])
			# after fitting all k's for all splits, determine best number of events:
			ROIsHMM[task]['best_tune_ll'] = np.argmax(np.mean(ROIsHMM[task]['tune_ll'],axis=0))
			ROIsHMM[task]['best_corr'] = []
			for wi,w in enumerate(win_range):
				ROIsHMM[task]['best_corr'].append(np.argmax(np.mean(ROIsHMM[task]['within_r'][:,:,wi]-ROIsHMM[task]['across_r'][:,:,wi],axis=0)))
		dd.io.save(ROIf,ROIsHMM)