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
grey = 211/256 # other options: https://www.rapidtables.com/web/color/gray-color.html

task='DM'
nTR_ = nTR[0]
for roi in tqdm.tqdm(glob.glob(HMMdir+'*.h5')):
	roi_short = roi.split('/')[-1][:-3]
	HMMtask = dd.io.load(roi,'/'+task)
	nshuff = len([k for k in list(HMMtask.keys()) if 'shuff' in k]) - 1
	p_auc = np.sum(abs(HMMtask['shuff_0']['auc_diff'])<[abs(HMMtask['shuff_'+str(s)]['auc_diff']) for s in range(1,nshuff+1)])/nshuff
	if p_auc<0.05 or 'LH_DefaultB_PFCd_1' in roi:
		time = np.arange(TR,nTR_*TR+1,TR)[:-1]
		k = HMMtask['best_k']
		D = [HMMtask['bin_'+str(b)]['D'] for b in bins]
		hmm = brainiak.eventseg.event.EventSegment(n_events=k)
		bin_tmp = bins if 'all' in HMMdir else [0,4]
		hmm.fit([np.mean(d,axis=0).T for d in [D[bi] for bi in bin_tmp]])
		kl = np.arange(k)+1
		fig, ax = plt.subplots(figsize=(10, 10))
		ax.set_xticks(np.append(time[0::nTR_//5],time[-1]))
		ax.set_xticklabels([str(int(s//60))+':'+str(int(s%60)) for s in time][0::nTR_//5]+['10:00'], fontsize=30)
		ax.set_xlabel('Time (seconds)', fontsize=35)
		klax = kl if k < 25 else kl[4::5]
		ax.set_yticks(klax)
		ax.set_yticklabels(klax,fontsize=30)
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
			ax.plot(time, E_k[bi], linewidth=5.5, alpha=0.5, color=colors[bi],label=lgd[bi])
		if 'LH_DefaultB_PFCd_1' in roi:
			ax.legend(prop={'size': 30},facecolor=(grey,grey,grey),edgecolor='black')
			plt.savefig(figurepath+'HMM/auc_FLUX/legend.png', bbox_inches='tight')
		else:
			ax.set_facecolor((grey,grey,grey))
			plt.savefig(figurepath+'HMM/auc_FLUX/'+roi_short+'.png', bbox_inches='tight')
			
#roi_example
# display low vs high ll
# display idealized squiggle plot
k = 5
kl = np.arange(k)+1
time = np.arange(TR,nTR_*TR+1,TR)[:-1]
ev_durr = [200,100,175,50,225]
even = [nTR_//k]*k
good_ll_fit = np.concatenate([i*np.ones(ev_durr[i-1]) for i in np.arange(1,k+1)],axis=0)
for win in [10,50,100]:
	inbet_ll = np.concatenate([good_ll_fit[:win],np.convolve(good_ll_fit, np.blackman(win)/np.sum(np.blackman(win)), 'same')[win:-win], good_ll_fit[-win:]])
	fig, ax = plt.subplots(figsize=(10, 10))
	ax.set_xticks(np.append(time[0::nTR_//5],time[-1]))
	ax.set_xticklabels([str(int(s//60))+':'+str(int(s%60)) for s in time][0::nTR_//5]+['10:00'], fontsize=30)
	ax.set_xlabel('Time in Movie (seconds)', fontsize=35)
	klax = kl if k < 25 else kl[4::5]
	ax.set_yticks(klax)
	ax.set_yticklabels(klax,fontsize=30)
	ax.set_ylabel('Events', fontsize=45)
	ax.set_facecolor((grey,grey,grey))
	ax.plot(time, inbet_ll, linewidth=5.5, alpha=0.5, color=colors[3],label=lgd[3])
	plt.savefig(figurepath+'HMM/auc_FLUX/'+str(win)+'.png', bbox_inches='tight')
	

colorinv = colors[::-1]
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xticks(np.append(time[0::nTR_//5],time[-1]))
ax.set_xticklabels([str(int(s//60))+':'+str(int(s%60)) for s in time][0::nTR_//5]+['10:00'], fontsize=30)
ax.set_xlabel('Time in Movie (seconds)', fontsize=35)
klax = kl if k < 25 else kl[4::5]
ax.set_yticks(klax)
ax.set_yticklabels(klax,fontsize=30)
ax.set_ylabel('Events', fontsize=45)
ax.set_facecolor((grey,grey,grey))
Ek = []
for b in bins:
	delay = b*np.random.randint(1,20)
	win = np.random.randint(1,30)
	inbet_ll = np.concatenate([good_ll_fit[:win],np.convolve(good_ll_fit, np.blackman(win)/np.sum(np.blackman(win)), 'same')[win:-win], good_ll_fit[-win:]])
	Ek = np.concatenate([np.ones(delay+10),inbet_ll[:-delay][10:]]) if delay>0 else inbet_ll
	ax.plot(time, Ek, linewidth=5.5, alpha=0.5, color=colorinv[b])
plt.savefig(figurepath+'HMM/auc_FLUX/'+'delay'+'.png', bbox_inches='tight')

	
	





