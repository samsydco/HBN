#!/usr/bin/env python3

# Log-likelihood plots

import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
from HMM_settings import *

HMMdir = HMMpath+'shuff_5bins_train04/'
figdir = figurepath + 'HMM/Paper/'
bins = np.arange(nbinseq)
nbins = len(bins)
lgd = [str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))+' y.o.' for b in bins]
colors = ['#edf8fb','#b3cde3','#8c96c6','#8856a7','#810f7c']
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
	legend = ax.legend(loc='lower right', bbox_to_anchor=(1.5, 0))
	fig.tight_layout()
	fig.savefig(figdir+task+'_'+roi+'.png', bbox_extra_artists=(legend,), bbox_inches='tight')
			
			
		
	
	