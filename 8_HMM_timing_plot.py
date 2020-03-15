#!/usr/bin/env python3

# Make Caroline-esque plots of E[k] x T (comparing young and old per ROI)

import glob
import tqdm
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from HMM_settings import *

ROInow = ROIopts[1]
HMMf = HMMpath+'timing_'+ROInow+'/'
ROIs = glob.glob(HMMf+'*h5')

yvsospred = {key: {} for key in tasks}
for roi in tqdm.tqdm(ROIs):
	roi_short = roi.split('/')[-1][:-3]
	ROIsHMM = dd.io.load(roi)
	for ti,task in enumerate(tasks):
		nTR_ = nTR[ti]
		yvsospred[task][roi_short] = {key: {} for key in ['best_tune_ll','best_corr']}
		time = np.arange(TR,nTR_*TR+1,TR)[:-1]
		yvsospred[task][roi_short]['best_tune_ll']['ll'] = np.max(np.mean(ROIsHMM[task]['tune_ll'],axis=0))
		yvsospred[task][roi_short]['best_tune_ll']['k'] = k_list[ROIsHMM[task]['best_tune_ll']]
		yvsospred[task][roi_short]['best_corr']['k'] = k_list[ROIsHMM[task]['best_corr']]
		D = [np.mean(ROIsHMM[task]['bin_0']['D'],axis=0).T,
			 np.mean(ROIsHMM[task]['bin_4']['D'],axis=0).T]
		for kstr,k in yvsospred[task][roi_short].items():
			kl = np.arange(k['k'])+1
			hmm = brainiak.eventseg.event.EventSegment(n_events=k['k'])
			hmm.fit(D)
			fig, ax = plt.subplots(figsize=(10, 10))
			ax.set_title(roi_short+' '+kstr+' '+task, fontsize=50)
			ax.set_xticks(time[0::nTR_//5])
			ax.set_xticklabels([str(int(s//60))+':'+str(int(s%60)) for s in time][0::nTR_//5], fontsize=30)
			ax.set_xlabel('Time (seconds)', fontsize=35)
			ax.set_yticks(kl)
			ax.set_yticklabels(kl,fontsize=30)
			ax.set_ylabel('Events', fontsize=45)
			E_k = []
			auc = []
			for bi in range(len(hmm.segments_)):
				E_k.append(np.dot(hmm.segments_[bi], kl))
				auc.append(round(E_k[bi].sum(), 2))
				ax.plot(time, E_k[bi], linewidth=5.5, alpha=0.5)
				ax.legend(['Young', 'Old'], fontsize=30)
			ax.fill_between(time, E_k[1], E_k[0],facecolor='silver', alpha=0.5)
			ax.text(time[-1], 2, 'Avg prediction = ',verticalalignment='bottom', horizontalalignment='right', fontsize=35)
			yvsospred[task][roi_short][kstr]['auc_diff'] = round((auc[1]-auc[0])/(k['k'])*TR, 2)
			ax.text(time[-1]-10, 1, str(round((auc[1]-auc[0])/(k['k'])*TR, 2)) + ' seconds', verticalalignment='bottom', horizontalalignment='right', fontsize=35)
			#plt.show()
			plt.savefig(figurepath+'HMM/timing/'+roi_short+'_'+kstr+'_'+task+'.png', bbox_inches='tight')

def extend_for_TP(array,task):
	xnans = np.nan*np.zeros(4)
	if task == 'DM':
		array = array
	else:
		array = np.concatenate((array,xnans), axis=0)
	return array
	
# Now plotting difference in tune_ll between young and old:
for roi in tqdm.tqdm(ROIs):
	roi_short = roi.split('/')[-1][:-3]
	ROIsHMM = dd.io.load(roi)
	fig,ax = plt.subplots(2,sharex=True)
	axa = fig.add_subplot(111, frameon=False)
	axa.set_title(roi_short)
	axa.set_xlabel('Average Event Duration',labelpad=20)
	axa.set_ylabel('Log likelihood\nPer TR',labelpad=40)
	axa.set_yticks([],[])
	axa.set_xticks([],[])
	for ti,task in enumerate(tasks):
		nTR_ = nTR[ti]
		x_list = [np.round(TR*(nTR_/k),2) for k in k_list]
		if task == 'TP': x_list = x_list+[120,150,200,300]; ax[ti].set_xticks(x_list,[]); ax[ti].set_xticklabels([])
		ax[ti].set_title(task)
		for b in bins:
			yvsospred[task][roi_short]['bin_'+str(b)] = {}
			# save both ll diff using r.h.s. baseline, and ll diff at max-ll k
			yvsospred[task][roi_short]['bin_'+str(b)]['tune_ll'] = np.mean(ROIsHMM[task]['bin_'+str(b)]['tune_ll'],0)/nTR_
			yvsospred[task][roi_short]['bin_'+str(b)]['max_ll_ll'] = yvsospred[task][roi_short]['bin_'+str(b)]['tune_ll'] \
												[np.where(k_list==yvsospred[task][roi_short]['best_tune_ll']['k'])[0][0]]
			c = '#1f77b4' if b == 0 else '#ff7f0e'
			lab = 'Ages '+str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))
			y = extend_for_TP(np.mean(ROIsHMM[task]['bin_'+str(b)]['tune_ll'],0)/nTR_,task)
			yerr = extend_for_TP(np.std(ROIsHMM[task]['bin_'+str(b)]['tune_ll'],0)/nTR_,task)
			ax[ti].errorbar(x_list, y, yerr=yerr, color=c, label=lab)	
	ax[ti].set_xticklabels(x_list,rotation=45)
	lgd = ax[ti].legend(loc='lower right', bbox_to_anchor=(1.3, 0))
	fig.set_size_inches(9,6)
	fig.tight_layout()
	#plt.show()
	fig.savefig(figurepath+'HMM/tune_ll_timing/'+roi_short+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	

# check that r.h.s. of ll's are aproximately equal across tune_ll, and each bin's tune_ll:
# It looks like avg. of 2 bin's r.h.s. ll's will be approximately equal
ROIshortl = list(wtf[task].keys())
tune_ll = {key: {key: {} for key in ROIshortl} for key in tasks}
bin_ll = {key: {key: {key: {} for key in ROIshortl} for key in tasks} for key in ['bin_0','bin_4']}
for roi in tqdm.tqdm(ROIs):
	roi_short = roi.split('/')[-1][:-3]
	ROIsHMM = dd.io.load(roi)
	for ti,task in enumerate(tasks):
		nTR_ = nTR[ti]
		tune_ll[task][roi_short] = np.mean(ROIsHMM[task]['tune_ll'],axis=0)[-1]/nTR_
		for b in bins:
			bin_ll['bin_'+str(b)][task][roi_short] = np.mean(ROIsHMM[task]['bin_'+str(b)]['tune_ll'],0)[-1]/nTR_

fig,ax = plt.subplots(2,sharex=True)
axa = fig.add_subplot(111, frameon=False)
axa.set_xlabel('ROI label',labelpad=20)
axa.set_ylabel('Log likelihood\nPer TR',labelpad=40)
axa.set_yticks([],[])
axa.set_xticks([],[])
for ti,task in enumerate(tasks):
	nTR_ = nTR[ti]
	ys = {}
	for b in bins:
		c = '#1f77b4' if b == 0 else '#ff7f0e'
		lab = 'Ages '+str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))
		ys['bin_'+str(b)] = []
		for v in bin_ll['bin_'+str(b)][task].values():
			ys['bin_'+str(b)].append(v)
		ax[ti].plot(ys['bin_'+str(b)], color=c, label=lab)
	for roi in tqdm.tqdm(ROIs):
		roi_short = roi.split('/')[-1][:-3]
		ROIsHMM = dd.io.load(roi)
		rhs = np.mean([bin_ll['bin_0'][task][roi_short],bin_ll['bin_4'][task][roi_short]])
		for b in bins:
			yvsospred[task][roi_short]['bin_'+str(b)]['ll_rhs'] = np.sum(yvsospred[task][roi_short]['bin_'+str(b)]['tune_ll'][yvsospred[task][roi_short]['bin_'+str(b)]['tune_ll']>rhs]-rhs)
	ax[ti].set_title(task+', r = '+str(np.round(np.corrcoef(ys['bin_0'],ys['bin_4'])[0,1],2)))
	ys['all'] = []
	for v in tune_ll[task].values():
		ys['all'].append(v)
	ax[ti].plot(ys['all'], color='#008000', label='Avg both age groups')
ax[ti].set_xticklabels(ROIshortl,rotation=45)
lgd = ax[ti].legend(loc='lower right', bbox_to_anchor=(1.3, 0))
fig.set_size_inches(9,6)
fig.tight_layout()
#plt.show()
fig.savefig(figurepath+'HMM/tune_ll_timing/RHS_ll.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

dd.io.save(HMMpath+'f_timing_'+ROInow,yvsospred)
			
				
				
				
			
			
			
			