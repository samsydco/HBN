#!/usr/bin/env python3

# For each ROI:
# Subtract: max-ll - ll-at-2k

import glob
import tqdm
import numpy as np
import deepdish as dd
from scipy.stats import pearsonr
from HMM_settings import *

task= 'DM'
nTR = nTR[0]

lldict = {}
for seed in seeds:
	lldict[seed] = {}
	for roi in glob.glob(nkdir+seed+'/'+'*.h5'):
		roi_short = roi.split('/')[-1][:-3]
		lldict[seed][roi_short] = {}
		for b in [0,4]:
			ll_sep = dd.io.load(roi,'/'+str(b)+'/tune_ll')
			lldict[seed][roi_short][str(b)+'_ll_max'] = np.max(np.mean(ll_sep,axis=0))/nTR
			lldict[seed][roi_short][str(b)+'_ll_min'] = np.min(np.mean(ll_sep,axis=0))/nTR
			lldict[seed][roi_short][str(b)+'_ll_diff'] = \
			lldict[seed][roi_short][str(b)+'_ll_max'] - \
			lldict[seed][roi_short][str(b)+'_ll_min']
			lldict[seed][roi_short][str(b)+'_2k'] = np.mean(ll_sep[:,0],axis=0)/nTR
			lldict[seed][roi_short][str(b)+'_2k_diff'] = \
			lldict[seed][roi_short][str(b)+'_ll_max'] - \
			lldict[seed][roi_short][str(b)+'_2k']
			lldict[seed][roi_short][str(b)+'_k'] = dd.io.load(roi,'/'+str(b)+'/best_k')	
		
dd.io.save(llh5,lldict)
lldict = dd.io.load(llh5)

# eliminate parcels with ll-diff below threshold in BOTH young and old subjects:
comp = '_2k_diff'
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40})

def seeddictstodf(d,drop_cols=None):
	dfs = {}
	for seed in seeds:
		dfs[seed] = pd.DataFrame(d[seed]).T
		if drop_cols!=None:
			dfs[seed]=dfs[seed].drop(columns=drop_cols)
	df = dfs[seed].iloc[0:0,:].copy()
	for seed in seeds:
		df = pd.concat([df,dfs[seed]]).astype('float')
	df=df.groupby(df.index).mean()
	return df

drop_cols = ['0_ll_diff','4_ll_diff','0_ll_max', '4_ll_max','0_2k', '4_2k']
lldf = seeddictstodf(lldict,drop_cols)
lldf.to_csv(llcsv)

x = lldf['0'+comp]; y = lldf['4'+comp]
idx1 = np.intersect1d(np.where(y<ll_thresh)[0],np.where(x<ll_thresh)[0])
idx2 = np.unique(np.concatenate((np.where(y>ll_thresh)[0],np.where(x>ll_thresh)[0])))
x1 = x.iloc[idx1]; y1 = y.iloc[idx1]
x2 = x.iloc[idx2]; y2 = y.iloc[idx2]
fig, ax = plt.subplots(figsize=(15,15))
ax.scatter(x1,y1,c='k',s=100,alpha=0.25)
ax.scatter(x2,y2,c='k',s=100,alpha=0.25)
ax.plot([ll_thresh,0],[ll_thresh,ll_thresh], "k--",linewidth=5)
ax.plot([ll_thresh,ll_thresh],[0,ll_thresh], "k--",linewidth=5)
ylim = ax.get_ylim()[1]
ax.plot([0,ylim],[0,ylim], "k--",linewidth=5,alpha=0.75)
ax.set_xlim([0,ylim])
ax.set_ylim([0,ylim])
ax.set_xticks([0,0.01,0.02,0.03])
ax.set_yticks([0,0.01,0.02,0.03])
ax.set_xlabel('Youngest Model-Fit difference')
ax.set_ylabel('Oldest Model-Fit difference')
fig.savefig(figurepath+'HMM/ll/'+comp+'_'+str(ll_thresh)+'_outlier.png', bbox_inches='tight')


np.round(TR*(nTR/lldf['0_k'].iloc[idx2]),2)

r,p = pearsonr(lldf['0_k'].iloc[idx2],lldf['4_k'].iloc[idx2])
rms_diff = np.sqrt(np.square(np.round(TR*(nTR/lldf['0_k'].iloc[idx2]),2) - np.round(TR*(nTR/lldf['4_k'].iloc[idx2]),2)))
rms_mean = np.mean(rms_diff)
rms_std = np.std(rms_diff)
