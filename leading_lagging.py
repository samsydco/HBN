#!/usr/bin/env python3

import tqdm
from HMM_settings import *
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def lag_pearsonr(x, y, max_lags):
    """Compute lag correlation between x and y, up to max_lags
    Parameters
    ----------
    x : ndarray
        First array of values
    y : ndarray
        Second array of values
    max_lags: int
        Largest lag (must be less than half the length of shortest array)
    Returns
    -------
    ndarray
        Array of 1 + 2*max_lags lag correlations, for x left shifted by
        max_lags to x right shifted by max_lags
    """

    assert max_lags < min(len(x), len(y)) / 2, \
        "max_lags exceeds half the length of shortest array"

    assert len(x) == len(y), "array lengths are not equal"

    lag_corrs = np.full(1 + (max_lags * 2), np.nan)

    for i in range(max_lags + 1):

        # add correlations where x is shifted to the right
        lag_corrs[max_lags + i] = pearsonr(x[:len(x) - i], y[i:len(y)])[0]

        # add correlations where x is shifted to the left
        lag_corrs[max_lags - i] = pearsonr(x[i:len(x)], y[:len(y) - i])[0]

    return lag_corrs

def nearest_peak(v):
	"""Estimates location of local maximum nearest the origin
    Starting at the origin, we follow the local gradient until reaching a
    local maximum. A quadratic function is then fit to the maximum and its
    two surrounding points, and the peak of this function is used as a
    continuous-valued estimate of the location of the maximum.
    Parameters
    ----------
    v : ndarray
        Array of values from [-max_lag, max_lag] inclusive
    Returns
    -------
    float
        Location of peak of quadratic fit
    """
	lag = (len(v)-1)//2
	# Find local maximum
	while 2 <= lag <= (len(v) - 3):
		win = v[(lag-1):(lag+2)]
		if (win[1] > win[0]) and (win[1] > win[2]):
			break
		if win[0] > win[2]:
			lag -= 1
		else:
			lag += 1
	# Quadratic fit
	x = [lag-1, lag, lag+1]
	y = v[(lag-1):(lag+2)]
	denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
	A = (x[2] * (y[1] - y[0]) + x[1] * \
			 (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
	B = (x[2]*x[2] * (y[0] - y[1]) + x[1]*x[1] * (y[2] - y[0]) + \
	x[0]*x[0] * (y[1] - y[2])) / denom
	max_x = (-B / (2*A))
	return min(max(max_x, 0), len(v)-1)

loaddir = HMMpath+'shuff_5bins_train04_'
bins = [0,4]
nTR = 750
max_lag = 25
#xcorrx = np.concatenate([np.arange(-nTR+2,0)*TR,np.arange(nTR-1)*TR])#[725:775]

lags = {k:{k:[] for k in ROIl} for k in seeds}
ps   = {k:{k:[] for k in ROIl} for k in seeds}
for seed in seeds:
	for roi in tqdm.tqdm(ROIl):
		dE_k = np.diff(dd.io.load(loaddir+seed+'/'+roi+'.h5','/E_k'))
		nshuffle = dE_k.shape[0]
		lags[seed][roi] = np.zeros((2,nshuffle))
		for bi,b in enumerate(bins):
			for shuffle in range(nshuffle):
				corr = lag_pearsonr(dE_k[shuffle,b], ev_conv[1:], max_lag)
				lags[seed][roi][bi,shuffle] = TR*(max_lag - nearest_peak(corr))
		lagdiff = lags[seed][roi][1] - lags[seed][roi][0]
		lagdiff = (lags[seed][roi][1] - lags[seed][roi][0])*TR
		ps[seed][roi] = np.sum(abs(lagdiff[0])<abs(lagdiff[1:]))/nshuffle
dd.io.save(HMMpath+'Leading_lagging.h5',{'lags':lags,'ps':ps})

roidict = dd.io.load(pvals_file,'/roidict')

lagsall = {}
lagsdiff = {}
psall = []
ROIs = []
for roi in ROIl:
	if roidict[roi]['auc_diff']['q'] < 0.05:
		ROIs.append(roi)
		lagsall[roi] = np.mean([lags[seed][roi] for seed in seeds],0)
		#lagsdiff[roi] = lagsall[roi][1] - lagsall[roi][0]
		lagsdiff[roi] = (lagsall[roi][1] - lagsall[roi][0])*TR
		p = np.sum(abs(lagsdiff[roi][0])<abs(lagsdiff[roi][1:]))/len(lagsdiff[roi])
		if p == 0:
			psall.append(1/len(lagsdiff[roi]))
		else:
			psall.append(p)
			

		
qs = FDR_p(np.array(psall))
for ri,roi in enumerate(ROIs):
	if qs[ri] < 0.05:
		print(roi,'auc diff',roidict[roi]['auc_diff']['val'],'oldest lag',lagsall[roi][1,0],'youngest lag',lagsall[roi][0,0])
		
dd.io.save(HMMpath+'Leading_lagging.h5',{'lags':lags,'ps':ps,'lagsall':lagsall,'psall':psall,'qs':qs,'ROIs':ROIs})


# for each seed, for each roi, plot:
# AUC, dE_k for oldest and yougest, and event annotations, xcorr,
# permuted differences, and real difference
event_annotations = ev_conv[1:]
plt.rcParams.update({'font.size': 30})
for roi in tqdm.tqdm(ROIs):
	fig,axs = plt.subplots(3,len(seeds),figsize=(60,20))
	fig.suptitle(roi)
	for si,seed in enumerate(seeds):
		AUC_diff = dd.io.load(HMMpath+'shuff_5bins_train04_'+seed+'/'+roi+'.h5','/'+'auc_diff')[0]
		s = 'Seed '+seed+' '+' AUC diff: '+str(np.round(AUC_diff,2))
		dE_k = np.diff(dd.io.load(loaddir+seed+'/'+roi+'.h5','/E_k'))[0]
		corrs = np.zeros((2,max_lag*2+1))
		for bi,b in enumerate(bins):
			corrs[bi] = lag_pearsonr(dE_k[b], ev_conv[1:], max_lag)
		lagdiff = (lags[seed][roi][1] - lags[seed][roi][0])*TR
		
		# Plot dE_k for oldest/youngest + event annotations
		ax2 = axs[0,si].twinx()
		l1= axs[0,si].plot(dE_k[0].T,color='g',label='Youngest dE_k')
		l2 = axs[0,si].plot(dE_k[4].T,color='b',label='Oldest dE_k')
		l3 = ax2.plot(ev_conv[1:],color='r',label='Event Annotations')
		lines = l1+l2+l3
		axs[0,si].set_xlabel('TR')
		axs[0,si].legend(lines,[l.get_label() for l in lines])
		axs[0,si].set_title(s)
		# Plot xcorr at various lags
		axs[1,si].plot(corrs.T)
		axs[1,si].set_xticks(np.arange(0,51,5))
		axs[1,si].set_xticklabels(np.concatenate((np.arange(max_lag, 0, -5), np.arange(0, max_lag+1, 5))))
		axs[1,si].legend(['Youngest correlations','Oldest correlations'])
		axs[1,si].set_xlabel('lag')
		axs[1,si].set_ylabel('correlation')
		# Plot perm diff vs real diff
		axs[2,si].hist(lagdiff[1:],bins=50,label='Permuted difference')
		axs[2,si].axvline(x=lagdiff[0], color='r', linestyle='--',label='Real difference')
		axs[2,si].legend()
		axs[2,si].set_ylabel('Count')
		fig.tight_layout(rect=[0, 0.03, 1, 0.95])
		fig.savefig(figurepath+'leading_lagging/'+roi+'.png')
		


