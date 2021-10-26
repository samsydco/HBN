#!/usr/bin/env python3

# Make Caroline plots in Yeo ROIs
# Make example plots

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import tqdm
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
import matplotlib.pyplot as plt
from HMM_settings import *

figdir = figurepath + 'HMM/Paper_auc/'
bins = np.arange(nbinseq)
nbins = len(bins)
xticks = [str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))+' y.o.' for b in bins]
colors = ['#FCC3A1','#F08B63','#D02941','#70215D','#311638']
plt.rcParams.update({'font.size': 30})

pvals = dd.io.load(pvals_file)
ROIl = []
for roi in pvals['roidict'].keys():
	if 'auc_diff' in pvals['roidict'][roi].keys():
		if pvals['roidict'][roi]['auc_diff']['q'] < 0.05:
			ROIl.append(roi)

allauc = np.zeros((len(ROIl),nbins))			
for ri,roi in enumerate(ROIl):
	vall = pvals['seeddict']['0'][roi]['vall']
	AUC = np.zeros((len(seeds),nbins))
	for si,seed in enumerate(seeds):
		k = dd.io.load(HMMsavedir+seed+'/'+roi+'.h5','/best_k')
		auc = (dd.io.load(HMMsavedir+seed+'/'+roi+'.h5','/auc')/(k-1))*TR
		for b in bins:
			AUC[si,b] = auc[0,b]
	AUC = np.mean(AUC,axis=0)
	allauc[ri] = AUC-AUC[0]
	
for ri,roi in enumerate(ROIl):	
	fig,ax = plt.subplots()
	ax.plot(np.arange(len(xticks)),allauc[ri], linestyle='-', marker='o', color='k')
	ax.set_xticks(np.arange(len(xticks)))
	ax.set_xticklabels(xticks,rotation=45, fontsize=20)
	ax.set_xlabel('Age',fontsize=20)
	ax.set_ylabel('Average Prediction Difference',fontsize=20)
	ax.set_ylim([np.min(allauc),np.max(allauc)])
	plt.show()
	fig.savefig(figdir+roi+'.png', bbox_inches="tight")

			




