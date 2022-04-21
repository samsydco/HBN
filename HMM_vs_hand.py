#!/usr/bin/env python3

# Compare event boundary timing in HMMs from cortical Yeo ROIs
# to timing in hand(RA)-labeled events

import os
import tqdm
import brainiak.eventseg.event
from scipy.fftpack import fft,ifft
from scipy.stats import zscore, norm, pearsonr
from HMM_settings import *
from event_comp import ev_conv, Pro_ev_conv, child_ev_conv

ev_conv = child_ev_conv

ev_conv_perm = ev_conv[1:]

task='DM'
nTR=750
nbins = len(bins)
nROI = len(ROIl)
xcorrx = np.concatenate([np.arange(-nTR+2,0)*TR,np.arange(nTR-1)*TR])
savefile = HMMpath+'HMM_vs_hand_child_'

dE_k = {key:{key:[] for key in bins} for key in ROIl}
dE_k_corr = np.zeros((nROI,nbins))
bin_corr = np.zeros(nROI)
#dE_k_p = np.zeros((nPerm+1,nROI,nbins))
event_bounds = {key:{key:[] for key in bins} for key in ROIl}
matchz_mat = np.zeros((nROI,nbins))

for seed in tqdm.tqdm(seeds):
	for r,roi_short in tqdm.tqdm(enumerate(ROIl)):
		roi=HMMsavedir+seed+'/'+roi_short+'.h5'
		k = dd.io.load(roi,'/best_k')
		D = [dd.io.load(roidir+seed+'/'+roi_short+'.h5','/'+task+'/bin_'+str(b)+'/D') for b in bins]
		hmm = brainiak.eventseg.event.EventSegment(n_events=k)
		hmm.fit([np.mean(d,axis=0).T for d in D])
		for bi,b in enumerate(bins):
			dE_k[roi_short][b] = np.diff(np.dot(hmm.segments_[bi], np.arange(k)+1))
			dE_k_corr[r,bi],_ = pearsonr(dE_k[roi_short][b],ev_conv_perm)
		bin_corr[r],_ = pearsonr(dE_k[roi_short][0],dE_k[roi_short][4])
	dd.io.save(savefile+'_'+seed+'.h5',{'dE_k_corr':dE_k_corr, 'dE_k':dE_k, 'bin_corr':bin_corr})

