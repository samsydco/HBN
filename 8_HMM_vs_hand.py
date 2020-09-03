#!/usr/bin/env python3

# Compare event boundary timing in HMMs from cortical Yeo ROIs
# to timing in hand(RA)-labeled events

import os
import tqdm
import brainiak.eventseg.event
from scipy.fftpack import fft,ifft
from scipy.stats import zscore, norm, pearsonr
from HMM_settings import *
ev_conv_perm = ev_conv[1:]

ROIdir = HMMpath+'shuff_5bins_train04/'
task='DM'
nTR=750
bins = np.arange(nbinseq)
nbins = len(bins)
ROIl = [roi.split('/')[-1][:-3] for roi in glob.glob(ROIdir+'*')]
nROI = len(ROIl)

nsub = 41

circ = False#False#True # Doing circular time shift vs phase shuffle for significant correlation
savefile = HMMpath+'HMM_vs_hand.h5'
if circ == True: 
	savefile = savefile[:-3]+'_circ.h5'
	print('Circular Time Shuffle')
TW = 10
nPerm = 2000 if circ == False else len(ev_conv_perm)-TW*2
	
# shuffle phase:
def phase_shuff(signal):
	# Modified from brainiak.utils.utils.phase_randomize
	prng = np.random.RandomState(None)
	# for signals with odd number of time points only:
	pos_freq = np.arange(1, (signal.shape[0] - 1) // 2 + 1)
	neg_freq = np.arange(signal.shape[0] - 1, (signal.shape[0] - 1) // 2, -1)
	phase_shifts = (prng.rand(len(pos_freq)) * 2 * np.math.pi)
	fft_data = fft(signal)
	# Shift pos and neg frequencies symmetrically, to keep signal real
	fft_data[pos_freq] *= np.exp(1j * phase_shifts)
	fft_data[neg_freq] *= np.exp(-1j * phase_shifts)
	# Inverse FFT to put data back in time domain
	signal_shuff = np.real(ifft(fft_data))
	return signal_shuff
	

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

dE_k = {key:{key:[] for key in bins} for key in ROIl}
dE_k_corr = np.zeros((nPerm+1,nROI,nbins))
dE_k_p = np.zeros((nPerm+1,nROI,nbins))
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
		ev_conv = ev_conv_perm
		dE_k[roi][b] = np.diff(np.dot(hmm.segments_[b], np.arange(best_k)+1))
		if circ == False:
			for p in range(nPerm+1):
				dE_k_corr[p,ri,b],dE_k_p[p,ri,b] = pearsonr(dE_k[roi][b],ev_conv)
				# shuffle ev_conv phase, keep mag
				ev_conv = phase_shuff(ev_conv)
		else:
			for p in range(nPerm+1):
				dE_k_corr[p,ri,b],dE_k_p[p,ri,b] = pearsonr(dE_k[roi][b],ev_conv)
				ev_conv = np.concatenate((ev_conv[p+TW:],ev_conv[:p+TW]))
				
		event_bounds[roi][b] = np.where(np.diff(np.argmax(hmm.segments_[b], axis = 1)))[0]
		matchz_mat[ri,b] = match_z(event_bounds[roi][b],event_list,nTR)

dd.io.save(savefile,{'event_bounds':event_bounds, 'matchz_mat':matchz_mat, 'dE_k_corr':dE_k_corr, 'dE_k_p':dE_k_p, 'dE_k':dE_k})

event_bounds, matchz_mat, dE_k_corr, dE_k_p=dd.io.load(savefile,['/event_bounds', '/matchz_mat', '/dE_k_corr', '/dE_k_p'])
# which ROIs have significant correlation with ev_conv?
nPerm1 = dE_k_corr.shape[0]
nPerm2 = 1000
sig_corr = {}
dE_k_age_rs = {}
dE_k_age_change = {}
sig_change = {}
for ri,roi in tqdm.tqdm(enumerate(ROIl)):
	pval = np.sum(np.mean(dE_k_corr[0,ri,:]) < np.mean(dE_k_corr[1:,ri,:],axis=1))/nPerm1
	if pval < 0.05:
		sig_corr[roi] = {'r':np.mean(dE_k_corr[0,ri,:]),'p':pval}
		dE_k_age_rs[roi] = np.zeros((nPerm2+1,nbins))
		dE_k_age_change[roi] = np.zeros(nPerm2+1)
		roidict = dd.io.load(ROIdir+roi+'.h5','/'+task)
		best_k = roidict['best_k']
		Dall = np.concatenate([roidict['bin_'+str(b)]['D'] for b in bins],axis=0)
		binord = np.concatenate([[b]*nsub for b in range(nbins)])
		for p in range(nPerm2+1):
			D = [np.mean(Dall[binord==b],axis=0).T for b in bins]
			hmm = brainiak.eventseg.event.EventSegment(n_events=best_k)
			hmm.fit(D)
			for b in bins:
				# Compare derivative of E_k to ev_conv:
				ev_conv = ev_conv_perm
				dE_k = np.diff(np.dot(hmm.segments_[b], np.arange(best_k)+1))
				dE_k_age_rs[roi][p,b],_ = pearsonr(dE_k,ev_conv)
			dE_k_age_change[roi][p],_ = pearsonr(bins,dE_k_age_rs[roi][p])
			# shuffle binord:
			binord = np.random.permutation(binord)
		pval2 = np.sum(abs(dE_k_age_change[roi][0]) < abs(dE_k_age_change[roi][1:]))/nPerm2
		if pval2 < 0.05:
			sig_change[roi] = {'r':dE_k_age_change[roi][0],'p':pval2}
			
dd.io.save(savefile,{'event_bounds':event_bounds, 'matchz_mat':matchz_mat, 'dE_k_corr':dE_k_corr, 'dE_k_p':dE_k_p, 'sig_corr':sig_corr, 'dE_k_age_rs':dE_k_age_rs, 'dE_k_age_change':dE_k_age_change, 'sig_change':sig_change})
	
event_bounds, matchz_mat, dE_k_corr, dE_k_p, sig_corr, dE_k_age_rs, dE_k_age_change, sig_change = dd.io.load(savefile,['/event_bounds', '/matchz_mat', '/dE_k_corr', '/dE_k_p', '/sig_corr', '/dE_k_age_rs', '/dE_k_age_change', '/sig_change'])
#corrmean = np.mean(dE_k_corr,axis=1)

