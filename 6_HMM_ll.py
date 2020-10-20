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
		lldict[roi_short][str(b)+'_2k'] = np.mean(ll[b,:,0],axis=0)/nTR
		lldict[roi_short][str(b)+'_2k_diff'] = lldict[roi_short][str(b)+'_ll_max'] - lldict[roi_short][str(b)+'_2k']
		
dd.io.save(HMMpath+'ll_diff.h5',lldict)
lldict = dd.io.load(HMMpath+'ll_diff.h5')

import pandas as pd
df=pd.DataFrame(lldict).T
df=df.drop(columns=['0_worstk','4_worstk','0_ll_diff','4_ll_diff','0_bestk','4_bestk',
				   '0_ll_min_opt','4_ll_min_opt','0_ll_max','4_ll_max','0_2k','4_2k'])

import matplotlib.pyplot as plt
import scipy.stats as stats
plt.rcParams.update({'font.size': 40})
thresh = 0.005
for comp in ['_2k_diff']:#['_ll_diff','_ll_min_opt','_2k']:
	excllist=df[((df['0'+comp]<thresh) | (df['4'+comp]<thresh))].index.values.tolist()
	x = df['0'+comp]; y = df['4'+comp]
	idx1 = np.unique(np.concatenate((np.where(y<thresh)[0],np.where(x<thresh)[0])))
	idx2 = np.intersect1d(np.where(y>thresh)[0],np.where(x>thresh)[0])
	x1 = x.iloc[idx1]; y1 = y.iloc[idx1]
	x2 = x.iloc[idx2]; y2 = y.iloc[idx2]
	fig, ax = plt.subplots(figsize=(15,15))
	ax.scatter(x1,y1,s=100,alpha=0.25)
	ax.scatter(x2,y2,s=100,alpha=0.25)
	ax.set_xlim([np.min(x)-0.005,np.max(x)+0.005])
	ax.set_ylim([np.min(y)-0.005,np.max(y)+0.005])
	ax.set_xlabel('Youngest '+comp[1:])
	ax.set_ylabel('Oldest '+comp[1:])
	ax.set_title('Outlier: '+df.loc[df['4'+comp]==df['4'+comp].max()].index[0])
	fig.savefig(figurepath+'HMM/ll/'+comp[1:]+'.png', bbox_inches='tight')


