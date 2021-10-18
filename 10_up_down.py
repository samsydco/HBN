#!/usr/bin/env python3

# Up / Down ISC graph for ALL bins
# Like SfN Poster Figure 2
# But now for all ROIs in Yeo atlas

import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import deepdish as dd
from ISC_settings import *

roidir = ISCpath+'Yeo_parcellation_'
ISCdir = ISCpath+'shuff_Yeo_'
figdir = figurepath+'up_down/'
pvals = dd.io.load(pvals_file)

task = 'DM'
nsub = 40
n_time = 750
bins = np.arange(nbinseq)
nbins = len(bins)
seeds = pvals['seeddict'].keys()
subh = [[[],[]]]
subh[0][0] = np.concatenate((np.arange(0,minageeq[0]//2),
						  minageeq[0]+np.arange(0,minageeq[1]//2)))
subh[0][1] = np.concatenate((np.arange(minageeq[0]//2,minageeq[0]),
						  minageeq[0]+np.arange(minageeq[1]//2,minageeq[1])))

plt.rcParams.update({'font.size': 15})
xticks = [str(int(round(eqbins[i])))+\
		  ' - '+str(int(round(eqbins[i+1])))+' y.o.' for i in range(len(eqbins)-1)]


for roi in pvals['roidict'].keys():
	if pvals['roidict'][roi]['ISC_e']['q'] < 0.05:
		print(roi)
		vall = pvals['seeddict']['0'][roi]['vall']
		n_vox = len(vall)
		ISC_w = np.zeros((len(seeds),nbins,n_vox))
		for si,seed in tqdm.tqdm(enumerate(seeds)):
			for b in bins:
				D,Age,Sex = load_D(roidir+seed+'/'+roi+'.h5',task,[b])
				ISC_w_,_ = ISC_w_calc(D,n_vox,n_time,nsub,subh)
				ISC_w[si,b] = np.reshape(ISC_w_,n_vox)
		ISC_w = np.mean(ISC_w,axis=0)
		plt.rcParams.update({'font.size': 30})
		fig,ax = plt.subplots()
		ax.plot(np.arange(len(xticks)),np.mean(ISC_w,axis=1), linestyle='-', marker='o', color='k')
		#ax.axes.errorbar(np.arange(len(xticks)),
		#				 np.mean(ISC_w,axis=1), 
		#				 yerr = np.std(ISC_w,axis=1), 
		#				 xerr = None, ls='none',capsize=10, elinewidth=1,fmt='.k',
		#				 markeredgewidth=1) 
		ax.set_xticks(np.arange(len(xticks)))
		ax.set_xticklabels(xticks,rotation=45, fontsize=20)
		ax.set_xlabel('Age',fontsize=20)
		ax.set_ylabel('ISC',fontsize=20)
		plt.show()
		fig.savefig(figdir+roi+'.png', bbox_inches="tight")
		#fig.figure.savefig(figdir+roi_short+sig+'.eps')
		
		plt.rcParams.update({'font.size': 20})
		fig,ax = plt.subplots(figsize=(2, 4))
		parts = ax.violinplot(pvals['roidict'][roi]['ISC_e']['shuff'], showmeans=False, showmedians=False,showextrema=False)
		for pc in parts['bodies']:
			pc.set_facecolor('k')
			#pc.set_edgecolor('black')
			#pc.set_alpha(1)
		ax.scatter(1,pvals['roidict'][roi]['ISC_e']['val']*-1,color='k',s=80)
		ax.set_xticks([])
		ax.set_ylabel('ISC difference',fontsize=30)
		fig.savefig(figdir+roi+'_ISC_difference.png', bbox_inches="tight")
		
		
		
	
	
	

	


