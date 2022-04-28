#!/usr/bin/env python3

import tqdm
from HMM_settings import *
from event_comp import ev_conv, Pro_ev_conv, child_ev_conv, lag_pearsonr
import matplotlib.pyplot as plt

ev_conv = child_ev_conv

lead_lag_file = 'Leading_lagging_children.h5'

bins = [0,4]
nTR = 750
max_lag = 25

lags = {k:{k:[] for k in ROIl} for k in seeds}
ps   = {k:{k:[] for k in ROIl} for k in seeds}
for seed in seeds:
	for roi in tqdm.tqdm(ROIl):
		dE_k = np.diff(dd.io.load(HMMsavedir+seed+'/'+roi+'.h5','/E_k'))
		nshuffle = dE_k.shape[0]
		lags[seed][roi] = np.zeros((2,nshuffle))
		for bi,b in enumerate(bins):
			for shuffle in range(nshuffle):
				corr = lag_pearsonr(ev_conv[1:], dE_k[shuffle,b], max_lag)
				lags[seed][roi][bi,shuffle] = TR*(max_lag - nearest_peak(corr))
		lagdiff = lags[seed][roi][1] - lags[seed][roi][0]
		lagdiff = (lags[seed][roi][1] - lags[seed][roi][0])*TR
		ps[seed][roi] = np.sum(abs(lagdiff[0])<abs(lagdiff[1:]))/nshuffle
dd.io.save(HMMpath+lead_lag_file,{'lags':lags,'ps':ps})

roidict = dd.io.load(pvals_file,'/roidict')

lagsall = {}
lagsdiff = {}
psall = []
ROIs = []
for roi in ROIl:
	if roidict[roi]['auc_diff']['q'] < 0.05:
		ROIs.append(roi)
		arrs = [np.array(lags[seed][roi]) for seed in seeds]
		arr = np.ma.empty((2,np.max([i.shape[1] for i in arrs]),len(arrs)))
		arr.mask = True
		for idx, l in enumerate(arrs):
			arr[:,:l.shape[1],idx] = l
		lagsall[roi] = arr.mean(axis = -1)
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
		
dd.io.save(HMMpath+lead_lag_file,{'lags':lags,'ps':ps,'lagsall':lagsall,'psall':psall,'qs':qs,'ROIs':ROIs})


# for each seed, for each roi, plot:
# AUC, dE_k for oldest and yougest, and event annotations, xcorr,
# permuted differences, and real difference
event_annotations = ev_conv[1:]
plt.rcParams.update({'font.size': 30})
for roi in tqdm.tqdm(ROIs):
	fig,axs = plt.subplots(3,len(seeds),figsize=(60,20))
	fig.suptitle(roi)
	for si,seed in enumerate(seeds):
		AUC_diff = dd.io.load(HMMsavedir+seed+'/'+roi+'.h5','/'+'auc_diff')[0]
		s = 'Seed '+seed+' '+' AUC diff: '+str(np.round(AUC_diff,2))
		dE_k = np.diff(dd.io.load(HMMsavedir+seed+'/'+roi+'.h5','/E_k'))[0]
		corrs = np.zeros((2,max_lag*2+1))
		for bi,b in enumerate(bins):
			corrs[bi] = lag_pearsonr(ev_conv[1:], dE_k[b], max_lag)
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
		


