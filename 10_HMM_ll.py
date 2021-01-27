#!/usr/bin/env python3

# Log-likelihood plots

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import tqdm
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
import matplotlib.pyplot as plt
from HMM_settings import *

figdir = figurepath + 'HMM/Paper_ll/'
bins = np.arange(nbinseq)
nbins = len(bins)
lgd = [str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))+' y.o.' for b in bins]
colors = ['#edf8fb','#b3cde3','#8c96c6','#8856a7','#810f7c']
# plot ll only for DM
plt.rcParams.update({'font.size': 25})
nTR_=750
x_list = [np.round(TR*(nTR_/k),2) for k in k_list]
pvals = dd.io.load(ISCpath+'p_vals_seeds.h5')
roi_short_list = {}
for roi in pvals['roidict'].keys():
	if 'll_diff' in pvals['roidict'][roi].keys():
		if pvals['roidict'][roi]['ll_diff']['q'] < 0.05:
			vallist = [abs(pvals['seeddict'][seed][roi]['ll_diff']['val']) for seed in seeds]
			roi_short_list[roi] = seeds[np.argmax(vallist)]
grey = 211/256
	
if os.path.exists(figdir+'savedict.h5'):
	savedict = dd.io.load(figdir+'savedict.h5')
else:
	savedict = {k:[] for k in roi_short_list}
for roi,seed in tqdm.tqdm(roi_short_list.items()):
	if not os.path.exists(figdir+'savedict.h5'):
		tune_ll = np.zeros((nbins,nsplit,len(k_list)))
		D = [dd.io.load(roidir+seed+'/'+roi+'.h5','/DM/bin_'+str(b)+'/D') for b in bins]
		for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
			Dtrain = [np.mean(d[Ls[0]],axis=0).T for d in [D[bi] for bi in [0,4]]]
			Dtest  = [np.mean(d[Ls[1]],axis=0).T for d in D]
			for ki,k in enumerate(k_list):
				hmm = brainiak.eventseg.event.EventSegment(n_events=k)
				hmm.fit(Dtrain)
				for b in bins:
					_, tune_ll[b,split,ki] = hmm.find_events(Dtest[b])
	savedict[roi] = tune_ll
	
	fig,ax = plt.subplots(figsize=(19,7))
	ax.set_xlabel('Average Event Duration [s]',fontsize=30)
	ax.set_ylabel('Log likelihood',fontsize=30)
	for b in bins:
		lab = 'Ages '+str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))
		y_ = np.mean(savedict[roi][b],0)/nTR_
		yerr = np.std(savedict[roi][b],0)/nTR_
		ax.errorbar(x_list, y_, yerr=yerr,color=colors[b],label=lab)
	ax.set_facecolor((grey,grey,grey))
	ax.set_xticks(x_list[::2])
	ax.set_xticklabels(x_list[::2],rotation=45)
	legend = ax.legend(loc='lower right', bbox_to_anchor=(1.5, 0))
	fig.tight_layout()
	fig.savefig(figdir+roi+'.png', bbox_extra_artists=(legend,), bbox_inches='tight')
	
dd.io.save(figdir+'savedict.h5',savedict)
			
			
		
	
	