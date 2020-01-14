#!/usr/bin/env python3

# Compute HMM for young and old group
# Compare: number of segments and boudaries of segments
# Leave out some subjects for fitting
# Iterate over number of events

import os
import glob
import tqdm
import itertools
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
from HMM_settings import *
from random import randrange
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

ROIopts = ['YeoROIsforSRM_sel_2020-01-14.h5','YeoROIsforSRM_2020-01-03.h5','SfN_2019/ROIs_Fig3/Fig3_','g_diff/']
ROInow = ROIopts[0]
ROIfold = path+'ROIs/'+ROInow
HMMf = HMMpath+ROInow+'/'
if not os.path.exists(HMMf):
    os.makedirs(HMMf)
# ROIs = dd.io.load(HMMf) if os.path.exists(HMMf) else {}
	
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
			# For young/old group
			for b in bins:
				ROIsHMM[task]['bin_'+str(b)] = {}
				subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
				ROIsHMM[task]['bin_'+str(b)]['subl'] = subl
				nsub = len(subl)
				# Load data
				dtmp = dd.io.load(subl[0],['/'+task+'/'+hemi],sel=dd.aslice[vall,:])[0]
				vall = [v for i,v in enumerate(vall) if i not in np.where(np.isnan(dtmp[:,0]))[0]]
				ROIsHMM['vall'] = vall
				ROIsHMM['nvox'] = len(vall)
				nTR = dtmp.shape[1]
				D = np.empty((nsub,len(vall),nTR),dtype='float16')
				for sidx, sub in enumerate(subl):
					D[sidx,:,:] = dd.io.load(sub,['/'+task+'/'+hemi],sel=dd.aslice[vall,:])[0]
				ROIsHMM[task]['bin_'+str(b)]['D'] = D
				for split in range(nsplit):
					ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)] = {}
					LI,LO = next(kf.split(np.arange(nsub)))
					ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['LI'] = LI
					ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['LO'] = LO
					#LO = np.random.choice(nsub,round(nsub*.2),replace=False) # 20%
					Dtrain = D[LI]
					Dtest = D[LO]
				
					# Fit HMM with VxT data, leaving some subjects out
					# preallocate
					for k in k_list:
						ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)] = {}
						#fit HMM
						hmms_wb = brainiak.eventseg.event.EventSegment(n_events=k)
						hmms_wb.fit(np.mean(Dtrain,axis=0).T)
						ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['pattern']=hmms_wb.event_pat_
						ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['seg_og']=hmms_wb.segments_[0]
						ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['event_var']=hmms_wb.event_var_
						# predict the event boundaries for the test set
						hmm_bounds, tune_ll = hmms_wb.find_events(np.mean(Dtest, axis=0).T)
						ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['seg_lo']=hmm_bounds
						ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['tune_ll']=tune_ll[0]
						# ll when randomly re-order event patterns
						perm_ll = []
						if k < 10: # < 10 factorial permutations
							ps = list(itertools.permutations(range(k)))
							if len(ps) > nshuff: # too many iterations, only do nshuff max
								idx = np.random.choice(np.arange(1,len(ps)), nshuff, replace=False)
							else:
								idx = np.arange(1,len(ps))
						else:
							ps = [np.random.permutation(k) for p in range(nshuff)]
							idx = np.arange(nshuff)
						for p in idx:
							permpat = hmms_wb.event_pat_[:,ps[p]]
							hmm_perm = brainiak.eventseg.event.EventSegment(n_events=k)
							hmm_perm.event_var_ = hmms_wb.event_var_ # event variance?
							hmm_perm.set_event_patterns(permpat)
							_, p_ll = hmm_perm.find_events(np.mean(Dtest, axis=0).T)
							perm_ll.append(p_ll)
						ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['perm_ll']=perm_ll
						events = np.argmax(hmm_bounds, axis=1)
						_, event_lengths = np.unique(events, return_counts=True)
						# Save segments_[0] with boundary timings
						hmm_bounds = np.where(np.diff(events))[0]
						# window size for within vs across correlations
						for w in win_range: # windows in range 5 - 10 sec
							ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['w_'+str(w)] = {}
							corrs = np.zeros(nTR-w)
							for t in range(nTR-w):
								corrs[t] = pearsonr(np.mean(Dtest, axis=0)[:,t],\
													np.mean(Dtest, axis=0)[:,t+w])[0]
							# Test within minus across boudary pattern correlation with held-out subjects
							ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['w_'+str(w)]['within_r']=corrs[events[:-w] == events[w:]]
							ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['w_'+str(w)]['across_r']=corrs[events[:-w] != events[w:]]
							ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['w_'+str(w)]['perms'] = {}
							for p in range(nshuff):
								rand_events = np.sort([randrange(k) for t in range(nTR)])
								ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['w_'+str(w)]['perms']['within_r_'+str(p)]=corrs[rand_events[:-w] == rand_events[w:]]
								ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['w_'+str(w)]['perms']['across_r_'+str(p)]=corrs[rand_events[:-w] != rand_events[w:]]
								
		
			# Find young patterns in old group and old in young, calc ll
			for b in bins: # error on next line
				bo = bins[-1] if b == 0 else 0
				D = ROIsHMM[task]['bin_'+str(b)]['D']
				ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]
				mismatch_ll = np.zeros((nsplit,len(k_list)))
				for split in range(nsplit):
					# test on same subset as above
					Dtest = D[ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['LO']]
					for k in k_list:
						pat = ROIsHMM[task]['bin_'+str(bo)]['split_'+str(split)]['k_'+str(k)]['pattern']
						var = ROIsHMM[task]['bin_'+str(bo)]['split_'+str(split)]['k_'+str(k)]['event_var']
						hmm_perm = brainiak.eventseg.event.EventSegment(n_events=k)
						hmm_perm.event_var_ = var # event variance?
						hmm_perm.set_event_patterns(pat)
						_, p_ll = hmm_perm.find_events(np.mean(Dtest, axis=0).T)
						ROIsHMM[task]['bin_'+str(b)]['split_'+str(split)]['k_'+str(k)]['mismatch_ll']=p_ll[0]
		for b in bins:
			bin = 'bin_'+str(b)
			ROIsHMM[task][bin]['all_sub_events'] = {}
			D = ROIsHMM[task][bin]['D']
			# set final patterns and event timings for each k with all subjects:
			for k in k_list:
				ROIsHMM[task][bin]['all_sub_events']['k_'+str(k)] = {}
				#fit HMM
				hmms_wb = brainiak.eventseg.event.EventSegment(n_events=k)
				hmms_wb.fit(np.mean(D,axis=0).T)
				ROIsHMM[task][bin]['all_sub_events']['k_'+str(k)]['pattern']=hmms_wb.event_pat_
				ROIsHMM[task][bin]['all_sub_events']['k_'+str(k)]['seg_og']=hmms_wb.segments_[0]
				ROIsHMM[task][bin]['all_sub_events']['k_'+str(k)]['event_var']=hmms_wb.event_var_
		dd.io.save(ROIf,ROIsHMM)

				


