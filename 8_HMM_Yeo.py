#!/usr/bin/env python3

# Make Caroline plots in Yeo ROIs

import glob
import tqdm
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
import matplotlib.pyplot as plt
from HMM_settings import *

HMMdir = HMMpath+'shuff_5bins_train04/'#'shuff_5bins_trainall/'#'shuff_5bins_train04/'#'shuff_5bins/'
figdirend = HMMdir.split('/')[-2][5:]+'/'
bins = np.arange(nbinseq)
nbins = len(bins)
lgd = [str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))+' y.o.' for b in bins]
colors = ['#edf8fb','#b3cde3','#8c96c6','#8856a7','#810f7c']


for roi in tqdm.tqdm(glob.glob(HMMdir+'*.h5')):
	roi_short = roi.split('/')[-1][:-3]
	ROIsHMM = dd.io.load(roi)
	for ti,task in enumerate(tasks):
		nTR_ = nTR[ti]
		time = np.arange(TR,nTR_*TR+1,TR)[:-1]
		k = ROIsHMM[task]['best_k']
		D = [ROIsHMM[task]['bin_'+str(b)]['D'] for b in bins]
		hmm = brainiak.eventseg.event.EventSegment(n_events=k)
		bin_tmp = bins if 'all' in HMMdir else [0,4]
		hmm.fit([np.mean(d,axis=0).T for d in [D[bi] for bi in bin_tmp]])
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
			if 'train' in HMMdir:
				seg, _ = hmm.find_events(np.mean(D[bi],axis=0).T)
			else:
				seg = hmm.segments_[bi]
			E_k.append(np.dot(seg, kl))
			auc.append(round(E_k[bi].sum(), 2))
			ax.plot(time, E_k[bi], linewidth=5.5, alpha=0.5, color=colors[bi])
		avgpred = [str(round((auc[bi+1]-auc[bi])/(k)*TR, 2)) for bi in range(len(bins)-1)]
		lgd_tmp = [lgd[i]+'\nPred: '+avgpred[i-1]+' s' if i>0 else lgd[i] for i in range(len(bins))]
		ax.legend(lgd_tmp, fontsize=25)
		#ax.fill_between(time, E_k[1], E_k[0],facecolor='silver', alpha=0.5)
		#ax.text(time[-1], 2, 'Avg prediction = ',verticalalignment='bottom', horizontalalignment='right', fontsize=35)
		#ax.text(time[-1]-10, 1, str(round((auc[1]-auc[0])/(k)*TR, 2)) + ' seconds', verticalalignment='bottom', horizontalalignment='right', fontsize=35)
		plt.savefig(figurepath+'HMM/Yeo_Caroline'+figdirend+roi_short+'_'+task+'.png', bbox_inches='tight')
		
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
			lab = 'Ages '+str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))
			y = extend_for_TP(np.mean(ROIsHMM[task]['tune_ll'][bi],0)/nTR_,task)
			yerr = extend_for_TP(np.std(ROIsHMM[task]['tune_ll'][bi],0)/nTR_,task)
			ax[ti].errorbar(x_list, y, yerr=yerr,color=colors[bi],label=lab)
	ax[ti].set_xticklabels(x_list,rotation=45)
	lgd = ax[ti].legend(loc='lower right', bbox_to_anchor=(1.3, 0))
	fig.set_size_inches(9,6)
	fig.tight_layout()
	#plt.show()
	fig.savefig(figurepath+'HMM/tune_ll_Yeo'+figdirend+roi_short+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	
# plot ll only for DM
plt.rcParams.update({'font.size': 25})
task='DM'
nTR_=750
roi_short_list=['LH_SalVentAttnA_ParOper_1','LH_VisCent_Striate_1']#'LH_DefaultB_PFCv_2'
grey = 211/256
for roi in roi_short_list:
	HMMtask = dd.io.load(HMMdir+roi+'.h5','/'+task)
	fig,ax = plt.subplots(figsize=(19,7))
	ax.set_xlabel('Average Event Duration [s]',fontsize=30)
	ax.set_ylabel('Log likelihood',fontsize=30)
	x_list = [np.round(TR*(nTR_/k),2) for k in k_list]
	for b in bins:
		lab = 'Ages '+str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))
		y = np.mean(HMMtask['tune_ll'][b],0)/nTR_
		yerr = np.std(HMMtask['tune_ll'][b],0)/nTR_
		ax.errorbar(x_list, y, yerr=yerr,color=colors[b],label=lab)
	ax.set_facecolor((grey,grey,grey))
	ax.set_xticks(x_list[::2])
	ax.set_xticklabels(x_list[::2],rotation=45)
	lgd = ax.legend(loc='lower right', bbox_to_anchor=(1.5, 0))
	fig.tight_layout()
	fig.savefig(figurepath+'HMM/tune_ll_Yeo'+figdirend+task+'_'+roi+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	
plt.rcParams.update({'font.size': 20})
# plot age vs ll, only at best k
for ti,task in enumerate(['DM']):
	nTR_ = nTR[ti]	
	for roi in tqdm.tqdm(glob.glob(HMMdir+'*.h5')):
		roi_short = roi.split('/')[-1][:-3]
		HMMtask = dd.io.load(roi,'/'+task)
		nshuff = len([k for k in list(HMMtask.keys()) if 'shuff' in k]) - 1
		p_ll_ = np.sum(abs(HMMtask['shuff_0']['ll_diff'])<[abs(HMMtask['shuff_'+str(s)]['ll_diff']) for s in range(1,nshuff+1)])/nshuff
		if p_ll_<0.05:
			print(roi_short)
			nullmean = np.mean(HMMtask['tune_ll_perm'][1:])/nTR_
			nullstd = np.std(HMMtask['tune_ll_perm'][1:])/nTR_
			fig,ax = plt.subplots()
			ax.fill_between(np.arange(nbinseq+1)-0.5,
							nullmean-nullstd, nullmean+nullstd,
							alpha=0.2, edgecolor='none', facecolor='grey')
			ax.axes.errorbar(np.arange(len(lgd)),
						 np.mean(HMMtask['tune_ll_perm'][0],axis=1)/nTR_, 
						 yerr = np.std(HMMtask['tune_ll_perm'][0],axis=1)/nTR_, 
						 xerr = None, ls='none',capsize=10, elinewidth=1,fmt='.k',
						 markeredgewidth=1) 
			ax.set_xticks(np.arange(len(lgd)))
			ax.set_xticklabels(lgd,rotation=45, fontsize=20)
			ax.set_xlabel('Age',labelpad=20, fontsize=20)
			ax.set_ylabel('Log likelihood\nPer TR',labelpad=20, fontsize=20)
			plt.show()
			fig.savefig(figurepath+'HMM/ll_FLUX/'+roi_short+'.png', bbox_inches='tight')
			
			
		
	
	