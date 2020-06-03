#!/usr/bin/env python3

# HPC univariate bump?
# 1) Sanity Check: Measure ISC in HPC
# Does HPC ISC increase with age?
# 2) Is there an HPC bump in oldest group at event boundaries?

import matplotlib.pyplot as plt
from ISC_settings import *
from scipy.stats import zscore
event_list = [56,206,244,243,373,404,443,506,544]

# Only looking at oldest group rn
b = 4
task = 'DM'
n_time=750
nsub=41
TW = 30
TR=0.8


D = np.empty((nsub,n_time),dtype='float16')
subl = [hpcprepath+ageeq[i][1][b][idx].split('/')[-1] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
for sidx, sub in enumerate(subl):
	D[sidx] = np.mean(dd.io.load(sub,['/'+task+'/HPC'])[0],axis=0)
	
# ISC
groups = np.zeros((2,n_time),dtype='float16')
for h in [0,1]:
	group = np.zeros((n_time),dtype='float16')
	groupn = np.ones((n_time),dtype='int')*nsub
	for i in np.arange(0+nsub//2*h,nsub//2+nsub//2*h):
		group = np.nansum(np.stack((group,D[i])),axis=0)
	groups[h] = zscore(group/groupn)
ISC = np.sum(np.multiply(groups[0],groups[1]))/(n_time-1)

# Univariate bump
plt.rcParams.update({'font.size': 30})
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
x=np.arange(-1*TW//2,TW//2)*TR
bumps   = np.zeros((len(event_list),TW))
bumpstd = np.zeros((len(event_list),TW))
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xticks([-10,-5,0,5,10])
ax.set_xticklabels
ax.set_yticks([-.15,0,.15])
ax.set_xlabel('Seconds from boundary', fontsize=35)
ax.set_ylabel('Boundary-Triggered\nHippocampal Activity', fontsize=35)
ax.axvline(0, c='k', ls='--')
for ei,e in enumerate(event_list):
	#bumpi = zscore(D[:,e-TW//2:e+TW//2],axis=1)
	bumpi = D[:,e-TW//2:e+TW//2]
	bumps[ei]   = np.mean(bumpi,axis=0)
	bumpstd[ei] = np.std (bumpi,axis=0)
	lab = 'Event at '+str(np.round(e*TR,2))+' s'
	ax.errorbar(x, bumps[ei], yerr=bumpstd[ei],color=colors[ei],label=lab)
ax.errorbar(x, np.mean(bumps,axis=0), yerr=np.mean(bumpstd,axis=0),c='k',ls='--',label='Avg')
lgd = ax.legend(loc='lower right', bbox_to_anchor=(1.7, -0.7))
plt.savefig(figurepath+'HMM/HPC.png', bbox_inches='tight')








