#!/usr/bin/env python3

# Make Caroline plots in Yeo ROIs

import glob
import tqdm
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
import matplotlib.pyplot as plt
from HMM_settings import *

HMMdir = HMMpath+'shuff/'

for roi in tqdm.tqdm(glob.glob(HMMdir+'*.h5')):
	roi_short = roi.split('/')[-1][:-3]
	ROIsHMM = dd.io.load(roi)
	for ti,task in enumerate(tasks):
		nTR_ = nTR[ti]
		time = np.arange(TR,nTR_*TR+1,TR)[:-1]
		k = ROIsHMM[task]['best_k']
		D = [np.mean(ROIsHMM[task]['bin_0']['D'],axis=0).T,
			 np.mean(ROIsHMM[task]['bin_4']['D'],axis=0).T]
		hmm = brainiak.eventseg.event.EventSegment(n_events=k)
		hmm.fit(D)
		kl = np.arange(k)+1
		fig, ax = plt.subplots(figsize=(10, 10))
		ax.set_title(roi_short+' '+task, fontsize=50)
		ax.set_xticks(time[0::nTR_//5])
		ax.set_xticklabels([str(int(s//60))+':'+str(int(s%60)) for s in time][0::nTR_//5], fontsize=30)
		ax.set_xlabel('Time (seconds)', fontsize=35)
		ax.set_yticks(kl)
		ax.set_yticklabels(kl,fontsize=30)
		ax.set_ylabel('Events', fontsize=45)
		E_k = []
		auc = []
		for bi in range(len(bins)):
			E_k.append(np.dot(hmm.segments_[bi], kl))
			auc.append(round(E_k[bi].sum(), 2))
			ax.plot(time, E_k[bi], linewidth=5.5, alpha=0.5)
			ax.legend(['Young', 'Old'], fontsize=30)
		ax.fill_between(time, E_k[1], E_k[0],facecolor='silver', alpha=0.5)
		ax.text(time[-1], 2, 'Avg prediction = ',verticalalignment='bottom', horizontalalignment='right', fontsize=35)
		ax.text(time[-1]-10, 1, str(round((auc[1]-auc[0])/(k)*TR, 2)) + ' seconds', verticalalignment='bottom', horizontalalignment='right', fontsize=35)
		plt.savefig(figurepath+'HMM/Yeo_Caroline/'+roi_short+'_'+task+'.png', bbox_inches='tight')
		
def extend_for_TP(array,task):
	xnans = np.nan*np.zeros(4)
	if task == 'DM':
		array = array
	else:
		array = np.concatenate((array,xnans), axis=0)
	return array
		
# Now plotting difference in tune_ll between young and old:
for roi in tqdm.tqdm(glob.glob(HMMdir+'*.h5')):
	roi_short = roi.split('/')[-1][:-3]
	ROIsHMM = dd.io.load(roi)
	fig,ax = plt.subplots(2,sharex=True)
	axa = fig.add_subplot(111, frameon=False)
	axa.set_xlabel('Average Event Duration',labelpad=20)
	axa.set_ylabel('Log likelihood\nPer TR',labelpad=40)
	axa.set_yticks([],[])
	axa.set_xticks([],[])
	for ti,task in enumerate(tasks):
		nTR_ = nTR[ti]
		x_list = [np.round(TR*(nTR_/k),2) for k in k_list]
		if task == 'TP': x_list = x_list+[120,150,200,300]; ax[ti].set_xticks(x_list,[]); ax[ti].set_xticklabels([])
		ax[ti].set_title(task)
		for bi,b in enumerate(bins):
			c = '#1f77b4' if b == 0 else '#ff7f0e'
			lab = 'Ages '+str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))
			y = extend_for_TP(np.mean(ROIsHMM[task]['tune_ll'][bi],0)/nTR_,task)
			yerr = extend_for_TP(np.std(ROIsHMM[task]['tune_ll'][bi],0)/nTR_,task)
			ax[ti].errorbar(x_list, y, yerr=yerr, color=c, label=lab)	
	ax[ti].set_xticklabels(x_list,rotation=45)
	lgd = ax[ti].legend(loc='lower right', bbox_to_anchor=(1.3, 0))
	fig.set_size_inches(9,6)
	fig.tight_layout()
	#plt.show()
	fig.savefig(figurepath+'HMM/tune_ll_Yeo/'+roi_short+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	
	