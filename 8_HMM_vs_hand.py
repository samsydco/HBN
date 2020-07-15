#!/usr/bin/env python3

# Compare event boundary timing in HMMs from cortical Yeo ROIs
# to timing in hand(RA)-labeled events

import tqdm
import brainiak.eventseg.event
from scipy.stats import zscore, norm, pearsonr
from HMM_settings import *

ROIdir = HMMpath+'shuff_5bins_train04/'
task='DM'
nTR=750
bins = np.arange(nbinseq)
nbins = len(bins)
ROIl = [roi.split('/')[-1][:-3] for roi in glob.glob(ROIdir+'*')]
nROI = len(ROIl)

# From Chris:
# Computes fraction of "ground truth" bounds are covered by a set of proposed bounds
# Returns z score relative to a null distribution via permutation
def match_z(proposed_bounds, gt_bounds, num_TRs):
    nPerm = 1000
    threshold = 7 # s.t. boundaries within 6 sec. 
    np.random.seed(0)

    gt_lengths = np.diff(np.concatenate(([0],gt_bounds,[num_TRs])))
    match = np.zeros(nPerm + 1)
    for p in range(nPerm + 1):
        gt_bounds = np.cumsum(gt_lengths)[:-1]
        for b in gt_bounds:
            if np.any(np.abs(proposed_bounds - b) <= threshold):
                match[p] += 1
        match[p] /= len(gt_bounds)
        gt_lengths = np.random.permutation(gt_lengths)
    
    return (match[0]-np.mean(match[1:]))/np.std(match[1:])

dE_k_corr = np.zeros((nROI,nbins))
dE_k_p = np.zeros((nROI,nbins))
event_bounds = {key:{key:[] for key in bins} for key in ROIl}
matchz_mat = np.zeros((nROI,nbins))
for ri,roi in tqdm.tqdm(enumerate(ROIl)):
	roidict = dd.io.load(ROIdir+roi+'.h5','/'+task)
	best_k = roidict['best_k']
	D = [np.mean(roidict['bin_'+str(b)]['D'],axis=0).T for b in bins]
	hmm = brainiak.eventseg.event.EventSegment(n_events=best_k)
	hmm.fit(D)
	for b in bins:
		# Compare derivative of E_k to ev_conv:
		dE_k = np.diff(np.dot(hmm.segments_[b], np.arange(best_k)+1))
		dE_k_corr[ri,b],dE_k_p[ri,b] = pearsonr(dE_k,ev_conv[1:])
		event_bounds[roi][b] = np.where(np.diff(np.argmax(hmm.segments_[b], axis = 1)))[0]
		matchz_mat[ri,b] = match_z(event_bounds[roi][b],event_list,nTR)

dd.io.save(HMMpath+'HMM_vs_hand.h5',{'event_bounds':event_bounds, 'matchz_mat':matchz_mat, 'dE_k_corr':dE_k_corr, 'dE_k_p':dE_k_p})

