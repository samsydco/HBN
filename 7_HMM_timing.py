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
from scipy.stats import mode
from random import randrange
from sklearn.model_selection import KFold

ROInow = ROIopts[1]
HMMf = HMMpath+'timing_'+ROInow+'/'
ROIfold = path+'ROIs/'+ROInow
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
		for task in ROIs[roi]['tasks']:
			ROIsHMM[task] = {}
			for b in bins:
				ROIsHMM[task]['bin_'+str(b)] = {}
				subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
				ROIsHMM[task]['bin_'+str(b)]['subl'] = subl
				nsub = len(subl)
				# Load data
				nTR = dd.io.load(subl[0],['/'+task+'/'+hemi])[0].shape[1]
				D = np.empty((nsub,len(vall),nTR),dtype='float16')
				badvox = []
				for sidx, sub in enumerate(subl):
					D[sidx,:,:] = dd.io.load(sub,['/'+task+'/'+hemi],sel=dd.aslice[vall,:])[0]
					badvox.extend(np.where(np.isnan(D[sidx,:,0]))[0]) # Some subjects missing some voxels
				D = np.delete(D,badvox,1)
				vall = np.delete(vall,badvox)
				ROIsHMM['vall'] = vall
				ROIsHMM['nvox'] = len(vall)
				ROIsHMM[task]['bin_'+str(b)]['D'] = D
			# saveing measures of HMM fit for finding number of events
			ROIsHMM[task]['tune_ll'] = np.zeros((nsplit,len(k_list)))
			ROIsHMM[task]['within_r'] = np.zeros((nsplit,len(k_list),len(win_range)))
			ROIsHMM[task]['across_r'] = np.zeros((nsplit,len(k_list),len(win_range)))
			for split in range(nsplit):
				splitsrt = 'split_'+str(split)
				ROIsHMM[task][splitsrt] = {}
				LI,LO = next(kf.split(np.arange(nsub)))
				Dtrain = [np.mean(ROIsHMM[task]['bin_0']['D'][LI],axis=0).T,
						  np.mean(ROIsHMM[task]['bin_4']['D'][LI],axis=0).T]
				Dtest =  [np.mean(ROIsHMM[task]['bin_0']['D'][LO],axis=0).T,
						  np.mean(ROIsHMM[task]['bin_4']['D'][LO],axis=0).T]
				# Fit HMM with TxV data, leaving some subjects out
				for ki,k in enumerate(k_list):
					kstr = 'k_'+str(k)
					ROIsHMM[task][splitsrt][kstr] = {}
					hmm = brainiak.eventseg.event.EventSegment(n_events=k)
					hmm.fit(Dtrain)
					ROIsHMM[task][splitsrt][kstr]['pattern']=hmm.event_pat_
					ROIsHMM[task][splitsrt][kstr]['seg_og']=hmm.segments_
					ROIsHMM[task][splitsrt][kstr]['event_var']=hmm.event_var_
					ROIsHMM[task][splitsrt][kstr]['train_ll']=hmm.ll_
					# predict the event boundaries for the average test set
					hmm_bounds, tune_ll = hmm.find_events(np.mean(np.dstack(Dtest),axis=2))
					ROIsHMM[task][splitsrt][kstr]['seg_lo']=hmm_bounds
					ROIsHMM[task]['tune_ll'][split,ki]=tune_ll[0]
					# predict the event boundaries for each bin
					for bi,b in enumerate(bins):
						_, tune_ll = hmm.find_events(Dtest[bi])
						ROIsHMM[task][splitsrt][kstr]['bin_'+str(b)+'_tune_ll'] = tune_ll
					events = np.argmax(hmm_bounds, axis=1)
					_, event_lengths = np.unique(events, return_counts=True)
					hmm_bounds = np.where(np.diff(events))[0]
					# window size for within vs across correlations
					for wi,w in enumerate(win_range): # windows in range 5 - 10 sec
						corrs = np.zeros(nTR-w)
						for t in range(nTR-w):
							corrs[t] = pearsonr(Dtest[t],Dtest[t+w])[0]
						# Test within minus across boudary pattern correlation with held-out subjects
						ROIsHMM[task]['within_r'][split,ki,wi] = np.mean(corrs[events[:-w] == events[w:]])
						ROIsHMM[task]['across_r'][split,ki,wi] = np.mean(corrs[events[:-w] != events[w:]])
			# after fitting all k's for all splits, determine best number of events:
			ROIsHMM[task]['best_tune_ll'] = np.argmax(np.mean(ROIsHMM[task]['tune_ll'],axis=0))
			ROIsHMM[task]['best_corr'] = np.argmax(np.mean(np.nanmean(ROIsHMM[task]['within_r']-ROIsHMM[task]['across_r'],axis=0),axis=1))
		dd.io.save(ROIf,ROIsHMM)
	else: # Now we are adding ll's for bin_0 and bin_4 (both train and test)
		ROIsHMM = dd.io.load(ROIf)
		for task in ROIs[roi]['tasks']:
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
		dd.io.save(ROIf,ROIsHMM)
						
				
			
		

	
