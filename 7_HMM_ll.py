#!/usr/bin/env python3

# For each ROI:
# Subtract: max-ll - ll-at-2k

import glob
import tqdm
import numpy as np
import deepdish as dd
from HMM_settings import *

task= 'DM'
nTR = nTR[0]

lldict = {}
for roi in glob.glob(nkdir+'*.h5'):
	roi_short = roi.split('/')[-1][:-3]
	lldict[roi_short] = {}
	for b in [0,4]:
		ll_sep = dd.io.load(roi,'/'+str(b)+'/tune_ll')
		lldict[roi_short][str(b)+'_ll_max'] = np.max(np.mean(ll_sep[0],axis=0))/nTR
		lldict[roi_short][str(b)+'_ll_min'] = np.min(np.mean(ll_sep[0],axis=0))/nTR
		lldict[roi_short][str(b)+'_ll_diff'] = \
		lldict[roi_short][str(b)+'_ll_max'] - \
		lldict[roi_short][str(b)+'_ll_min']
		lldict[roi_short][str(b)+'_2k'] = np.mean(ll_sep[0,:,0],axis=0)/nTR
		lldict[roi_short][str(b)+'_2k_diff'] = \
		lldict[roi_short][str(b)+'_ll_max'] - \
		lldict[roi_short][str(b)+'_2k']
		
dd.io.save(llh5,lldict)
lldict = dd.io.load(llh5)

# eliminate parcels with ll-diff below threshold in BOTH young and old subjects:
comp = '_2k_diff'
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40})

df=pd.DataFrame(lldict).T
df=df.drop(columns=['0_ll_diff','4_ll_diff','0_ll_max','4_ll_max','0_2k','4_2k'])
x = df['0'+comp]; y = df['4'+comp]
idx1 = np.intersect1d(np.where(y<ll_thresh)[0],np.where(x<ll_thresh)[0])
idx2 = np.unique(np.concatenate((np.where(y>ll_thresh)[0],np.where(x>ll_thresh)[0])))
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
fig.savefig(figurepath+'HMM/ll/'+comp+'_'+str(ll_thresh)+'.png', bbox_inches='tight')


# In Parcels above minimum ll: is there a difference in number of k's (events)?
roidict=dd.io.load(nkh5)
df = pd.DataFrame(roidict).T.merge(pd.DataFrame(lldict).T, left_index=True, right_index=True, how='inner')
df=df[((df['0_2k_diff']>ll_thresh) | (df['4_2k_diff']>ll_thresh))]
df['k_diff_q'] = FDR_p(df['k_diff_p'])

from scipy.stats import pearsonr
import seaborn as sns
grey=211/256
xticks = [str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))+' y.o.' for b in bins]
sns.set(font_scale = 2,rc={'axes.facecolor':(grey,grey,grey)})
fig,ax=plt.subplots(figsize=(7,5))
sns.regplot(x='0',y='4',data=df,color='#8856a7',scatter_kws={'s':50})
ax.grid(False)
ax.set_xlabel('Number of events in\nYoungest ('+xticks[0]+')')
ax.set_ylabel('Number of events in\nOldest ('+xticks[1]+')')
ax.set(xlim=(6, 19),ylim=(6, 27.5))
print('r = '+str(np.round(r,2))+', p = '+str(np.round(p,8)))
fig.savefig(figurepath+'n_k/'+'k_lim.png',bbox_inches='tight', dpi=300)
