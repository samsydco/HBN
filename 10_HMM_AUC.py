#!/usr/bin/env python3

# Make Caroline plots in Yeo ROIs
# Make example plots

import glob
import tqdm
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
import matplotlib.pyplot as plt
from HMM_settings import *

HMMdir = HMMpath+'shuff_5bins_train04_paper/'
figdir = figurepath + 'HMM/Paper/'
bins = np.arange(nbinseq)
nbins = len(bins)
lgd = [str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))+' y.o.' for b in bins]
colors = ['#edf8fb','#b3cde3','#8c96c6','#8856a7','#810f7c']
grey = 211/256 # other options: https://www.rapidtables.com/web/color/gray-color.html

q_vals = dd.io.load(ISCpath+'p_vals_paper.h5')
qrois = []
for roi in q_vals:
	if 'auc_diff' in q_vals[roi].keys():
		if q_vals[roi]['auc_diff']['q'] < 0.05:
			qrois.append(roi)

ROIl = [HMMdir+roi+'.h5' for roi in qrois]

task='DM'
nTR_ = nTR[0]
time = np.arange(TR,nTR_*TR+1,TR)[:-1]
for roi in tqdm.tqdm(ROIl):
	roi_short = roi.split('/')[-1][:-3]
	k = dd.io.load(roi,'/'+task+'/best_k')
	D = [dd.io.load(roi,'/'+task+'/bin_'+str(b)+'/D') for b in bins]
	hmm = brainiak.eventseg.event.EventSegment(n_events=k)
	bin_tmp = bins if 'all' in HMMdir else [0,4]
	hmm.fit([np.mean(d,axis=0).T for d in [D[bi] for bi in bin_tmp]])
	kl = np.arange(k)+1
	klax = kl if k < 25 else kl[4::5]
	# start figure plotting:
	fig, ax = plt.subplots(figsize=(10, 10))
	ax.set_xticks(np.append(time[0::nTR_//5],time[-1]))
	ax.set_xticklabels([str(int(s//60))+':'+str(int(s%60))+'0' for s in time][0::nTR_//5]+['10:00'], fontsize=30)
	ax.set_xlabel('Time (minutes)', fontsize=35)
	ax.set_yticks(klax)
	ax.set_yticklabels(klax,fontsize=30)
	ax.set_ylabel('Events', fontsize=45)
	E_k = []
	auc = []
	for bi in range(len(bins)):
		seg, _ = hmm.find_events(np.mean(D[bi],axis=0).T)
		E_k.append(np.dot(seg, kl))
		auc.append(round(E_k[bi].sum(), 2))
		ax.plot(time, E_k[bi], linewidth=5.5, alpha=0.5, color=colors[bi],label=lgd[bi])
	legend = ax.legend(prop={'size': 30},bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,edgecolor='black')
	ax.set_facecolor((grey,grey,grey))
	plt.savefig(figdir+roi_short+'.png', bbox_extra_artists=(legend,), bbox_inches='tight')
	#zoomed figure
	for z in [0,200,400]:
		ztw = np.arange(z,z+200)
		timez=time[ztw]
		fig, ax = plt.subplots(figsize=(10, 10))
		ax.set_xticks(np.append(timez[0::nTR_//5],time[-1]))
		ax.set_xticklabels([str(int(s//60))+':'+str(int(s%60)) for s in timez][0::nTR_//5]+['10:00'], fontsize=30)
		ax.set_xlabel('Time (minutes)', fontsize=35)
		ax.set_yticks(klax)
		ax.set_yticklabels(klax,fontsize=30)
		ax.set_ylabel('Events', fontsize=45)
		for bi in range(len(bins)):
			ax.plot(timez, E_k[bi][ztw], linewidth=5.5, alpha=0.5, color=colors[bi],label=lgd[bi])
		ax.set_facecolor((grey,grey,grey))
		plt.savefig(figdir+roi_short+'_'+str(z)+'_zoom.png', bbox_inches='tight')
			
#roi_example
# display low vs high ll
# display idealized squiggle plot
k = 3
kl = np.arange(k)+1
#ev_durr = [200,100,175,50,225]
ev_durr = [75,50,75]
even = [nTR_//k]*k
good_ll_fit = np.concatenate([i*np.ones(ev_durr[i-1]) for i in np.arange(1,k+1)],axis=0)
for win in [10,50]:
	inbet_ll = np.convolve(good_ll_fit, np.blackman(win)/np.sum(np.blackman(win)))[win:200-win//2]
	fig, ax = plt.subplots(figsize=(10, 10))
	ax.set_xticks([])
	ax.set_xlabel('Time in Movie', fontsize=45,labelpad=20)
	klax = kl if k < 25 else kl[4::5]
	ax.set_yticks(klax)
	ax.set_yticklabels(klax,fontsize=45)
	ax.set_ylabel('Events', fontsize=45,labelpad=20)
	ax.set_facecolor((grey,grey,grey))
	ax.plot(inbet_ll, linewidth=9, alpha=0.5, color=colors[3])
	plt.savefig(figdir+'_'.join([str(k),str(win)])+'.png', bbox_inches='tight')
	

colorinv = colors[::-1]
labs = ['Young','Old']
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xticks([])
ax.set_xlabel('Time in Movie', fontsize=45,labelpad=20)
klax = kl if k < 25 else kl[4::5]
ax.set_yticks(klax)
ax.set_yticklabels(klax,fontsize=45)
ax.set_ylabel('Events', fontsize=45,labelpad=20)
ax.set_facecolor((grey,grey,grey))
Ek = []
win=10
for bi,b in enumerate([4,0]):
	delay = b*10
	inbet_ll = np.concatenate([good_ll_fit[:win],np.convolve(good_ll_fit, np.blackman(win)/np.sum(np.blackman(win)), 'same')[win:-win], good_ll_fit[-win:]])
	Ek = np.concatenate([np.ones(delay+10),inbet_ll[:-delay][10:]]) if delay>0 else inbet_ll
	ax.plot(Ek, linewidth=9, alpha=0.5, color=colorinv[b],label=labs[bi])
lgd = ax.legend(loc='upper left',prop={'size':40},facecolor=(grey,grey,grey),frameon = True,edgecolor='k')
lgd.get_frame().set_linewidth(2)
plt.savefig(figdir+str(k)+'_delay'+'.png', bbox_inches='tight')

	
	





