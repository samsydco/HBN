#!/usr/bin/env python3

# For each ROI:
# Subtract: max-ll - min-ll
# Look for where max-ll - min-ll > 0 or some threshold
# Also include: min(tune_ll) at optimal number of events

import glob
import tqdm
import numpy as np
import deepdish as dd
from HMM_settings import *

HMMdir = HMMpath+'shuff_5bins_train04/'
task= 'DM'
nTR = nTR[0]

lldict = {}
for roi in glob.glob(HMMdir+'*.h5'):
	roi_short = roi.split('/')[-1][:-3]
	lldict[roi_short] = {}
	ll = dd.io.load(roi,'/'+task+'/tune_ll')
	nbins,nsplit,nk = ll.shape
	for b in [0,4]:
		lldict[roi_short][str(b)+'_ll_max'] = np.max(np.mean(ll[b],axis=0))/nTR
		lldict[roi_short][str(b)+'_ll_min'] = np.min(np.mean(ll[b],axis=0))/nTR
		lldict[roi_short][str(b)+'_ll_diff'] = lldict[roi_short][str(b)+'_ll_max'] - lldict[roi_short][str(b)+'_ll_min']
		idx = np.argmax(np.mean(ll[b],axis=0))
		lldict[roi_short][str(b)+'_ll_min_opt'] = np.min(ll[b,:,idx])/nTR
		lldict[roi_short][str(b)+'_bestk'] = np.mean([k_list[np.argmax(ll[b,ki])] for ki in range(nsplit)])
		lldict[roi_short][str(b)+'_worstk'] = np.mean([k_list[np.argmin(ll[b,ki])] for ki in range(nsplit)])
		
dd.io.save(HMMpath+'ll_diff.h5',lldict)
lldict = dd.io.load(HMMpath+'ll_diff.h5')

import pandas as pd
df=pd.DataFrame(lldict).T

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40})
for comp in ['_ll_diff','_ll_min_opt']:
	x = df['0'+comp]; y = df['4'+comp]
	fig, ax = plt.subplots(figsize=(15,15))
	ax.scatter(x,y,s=100,alpha=0.25)
	ax.set_xlim([np.min(x)-0.005,np.max(x)+0.005])
	ax.set_ylim([np.min(y)-0.005,np.max(y)+0.005])
	ax.set_xlabel('Youngest '+comp[1:])
	ax.set_ylabel('Oldest '+comp[1:])
	ax.set_title('Outlier: '+df.loc[df['4'+comp]==df['4'+comp].max()].index[0])
	fig.savefig(figurepath+'HMM/ll/'+comp[1:]+'.png', bbox_inches='tight')


