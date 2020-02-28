#!/usr/bin/env python3

# Make Caroline-esque plots of E[k] x T (comparing young and old per ROI)

import glob
import tqdm
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
import matplotlib.pyplot as plt
from HMM_settings import *

ROIopts = ['YeoROIsforSRM_sel_2020-01-14.h5','YeoROIsforSRM_2020-01-03.h5','SfN_2019/ROIs_Fig3/Fig3_','g_diff/']
ROInow = ROIopts[1]
HMMf = HMMpath+'timing_'+ROInow+'/'
ROIs = glob.glob(HMMf+'*h5')

yvsospred = {key: {} for key in tasks}
for roi in tqdm.tqdm(ROIs):
	roi_short = roi.split('/')[-1][:-3]
	ROIsHMM = dd.io.load(roi)
	for ti,task in enumerate(tasks):
		yvsospred[task][roi_short] = {key: {} for key in ['best_tune_ll','best_corr']}
		time = np.arange(TR,nTR[ti]*TR+1,TR)[:-1]
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
			ax.set_xticks(time[0::nTR[ti]//5])
			ax.set_xticklabels([str(int(s//60))+':'+str(int(s%60)) for s in time][0::nTR[ti]//5], fontsize=30)
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

dd.io.save(HMMpath+'f_timing_'+ROInow,yvsospred)
			
				
				
				
			
			
			
			