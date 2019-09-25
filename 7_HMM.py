#!/usr/bin/env python3

# Compute HMM for young and old group
# Compare: number of segments and boudaries of segments
# Leave out some subjects for fitting
# Iterate over number of events

import os
import glob
import tqdm
import h5py
import itertools
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
from datetime import date
from HMM_settings import *
from random import randrange
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

ROIfold = path+'ROIs/g_diff/'
HMMf = HMMpath+'HMM_'+str(date.today())+'.h5'
if os.path.exists(HMMf):
    os.remove(HMMf)
	
kf = KFold(n_splits=nsplit,shuffle=True)

with h5py.File(HMMf,'a') as hf:
	for f in tqdm.tqdm(glob.glob(ROIfold+'*roi')):
		fn = f.split(ROIfold)[1]
		roin = fn[:-7]
		grpf = hf.create_group(roin)
		task = 'DM' if fn[:2] == 'TP' else 'TP'
		hemi = fn[3]
		vall = []
		with open(f, 'r') as inputfile:
			for line in inputfile:
				if len(line.split(' ')) == 3:
					vall.append(int(line.split(' ')[1]))
		n_vox = len(vall)
		grpf.create_dataset('vall',data=vall)
		# For young/old group
		for b in bins:
			grpb = grpf.create_group('bin_'+str(b))
			subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
			grpb.create_dataset('subl', (len(subl),1),'S48', [n.encode("ascii", "ignore") for n in subl])
			nsub = len(subl)
			# Load data
			_,n_time = dd.io.load(subl[0],['/'+task+'/'+hemi])[0].shape
			D = np.empty((len(subl),n_vox,n_time),dtype='float16')
			for sidx, sub in enumerate(subl):
				D[sidx,:,:] = dd.io.load(sub,['/'+task+'/'+hemi],sel=dd.aslice[vall,:])[0]
			grpb.create_dataset('D',data=D)
			for split in range(nsplit):
				grps = grpb.create_group('split_'+str(split))
				LI,LO = next(kf.split(np.arange(nsub)))
				grps.create_dataset('LI', data=LI)
				grps.create_dataset('LO', data=LO)
				#LO = np.random.choice(nsub,round(nsub*.2),replace=False) # 20%
				Dtrain = D[LI]
				Dtest = D[LO]
		
				# Fit HMM with VxT data, leaving some subjects out
				# preallocate
				for i,k in enumerate(k_list):
					grpk = grps.create_group('k_'+str(k))
					#fit HMM
					hmms_wb = brainiak.eventseg.event.EventSegment(n_events=k)
					hmms_wb.fit(np.mean(Dtrain,axis=0).T)
					grpk.create_dataset('pattern',data=hmms_wb.event_pat_)
					grpk.create_dataset('seg_og',data=hmms_wb.segments_[0])
					grpk.create_dataset('event_var',data=hmms_wb.event_var_)
					# predict the event boundaries for the test set
					hmm_bounds, tune_ll = hmms_wb.find_events(np.mean(Dtest, axis=0).T)
					grpk.create_dataset('seg_lo',data=hmm_bounds)
					grpk.create_dataset('tune_ll',data=tune_ll[0])
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
					grpk.create_dataset('perm_ll',data=perm_ll)
					events = np.argmax(hmm_bounds, axis=1)
					_, event_lengths = np.unique(events, return_counts=True)
					# Save segments_[0] with boundary timings
					hmm_bounds = np.where(np.diff(events))[0]
					# window size for within vs across correlations
					for w in win_range: # windows in range 5 - 10 sec
						grpw = grpk.create_group('w_'+str(w))
						corrs = np.zeros(nTR-w)
						for t in range(nTR-w):
							corrs[t] = pearsonr(np.mean(Dtest, axis=0)[:,t],\
												np.mean(Dtest, axis=0)[:,t+w])[0]
						# Test within minus across boudary pattern correlation with held-out subjects
						grpw.create_dataset('within_r',data=corrs[events[:-w] == events[w:]])
						grpw.create_dataset('across_r',data=corrs[events[:-w] != events[w:]])
						grpp = grpw.create_group('perms')
						for p in range(nshuff):
							rand_events = np.sort([randrange(k) for t in range(nTR)])
							grpp.create_dataset('within_r_'+str(p),data=corrs[rand_events[:-w] == rand_events[w:]])
							grpp.create_dataset('across_r_'+str(p),data=corrs[rand_events[:-w] != rand_events[w:]])
		# Find young patterns in old group and old in young, calc ll
		for b in bins: # error on next line
			bo = bins[-1] if b == 0 else 0
			D = dd.io.load(HMMf,'/'+roin+'/bin_'+str(b)+'/D')
			grpb = grpf.require_group('bin_'+str(b))
			mismatch_ll = np.zeros((nsplit,len(k_list)))
			for split in range(nsplit):
				grps = grpb.require_group('split_'+str(split))
				for i,k in enumerate(k_list):
					grpk = grps.require_group('k_'+str(k))
					pat = dd.io.load(HMMf,
									 '/'+roin+'/bin_'+str(bo)+'/split_'+str(split)+'/k_'+str(k)+'/pattern')
					var = dd.io.load(HMMf,
									 '/'+roin+'/bin_'+str(bo)+'/split_'+str(split)+'/k_'+str(k)+'/event_var')
					hmm_perm = brainiak.eventseg.event.EventSegment(n_events=k)
					hmm_perm.event_var_ = var # event variance?
					hmm_perm.set_event_patterns(pat)
					_, p_ll = hmm_perm.find_events(np.mean(D, axis=0).T)
					grpk.create_dataset('mismatch_ll',data=p_ll[0])
					

				



