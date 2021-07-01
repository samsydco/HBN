#!/usr/bin/env python3

import tqdm
import random
import numpy as np
import deepdish as dd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from ISC_settings import *

nseed = 5
nsub = 41
bins = [0,4]
task='DM'
n_time=750

# mrvect used to have a list of diff motion regressors
motion = 'FramewiseDisplacement'
anals = ['mean','isc_w']
mrdict = {a:{s:{s:[] for s in range(nshuff+1)} for s in range(nseed)} for a in anals+['vals']}

D = np.empty((nseed,nsub*2,n_time),dtype='float16')
for seed in range(nseed):
	Age = []
	Sex = []
	for bi,b in enumerate(bins):
		np.random.seed(seed)
		subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
		Sex.extend([Phenodf['Sex'][Phenodf['EID'] == shortsub(sub)].iloc[0] for sub in subl])
		Age.extend([bi]*len(subl))
		sub_ = 0 if b==0 else nsub # young and then old in D
		for sidx, sub in enumerate(subl):
			D[seed,sidx+sub_] = dd.io.load(sub,['/'+task+'/reg'])[0][:,2]
	for shuff in tqdm.tqdm(range(nshuff+1)):
		if shuff>0:
			Age,Sex = shuff_demo(shuff,Age,Sex)
		subh = even_out(Age,Sex)
		group = np.zeros((2,2,nsub//2,n_time))
		for h in [0,1]:
			for htmp in [0,1]:
				for i,sub in enumerate(subh[h][htmp]):
					group[h,htmp,i] = D[seed,sub]
		elimsub = np.where(np.isnan(group))[2]
		if len(elimsub)>0: elimsub=elimsub[0]
		for bi,b in enumerate(bins):
			g1 = np.delete(group[bi,0],elimsub,0)
			g2 = np.delete(group[bi,1],elimsub,0)
			vals = np.concatenate((g1,g2))
			mrdict['isc_w'][seed][shuff].append(pearsonr(\
				np.mean(g1,axis=0),\
				np.mean(g2,axis=0))[0])
			mrdict['mean'][seed][shuff].append(np.mean(vals))
			mrdict['vals'][seed][shuff].append(np.mean(vals,1))
		
colors = ['#1f77b4', '#ff7f0e']
plt.rcParams.update({'font.size': 22})

fig, axs = plt.subplots(len(anals), 1,figsize=(10, 10))
for a,anal in enumerate(anals):
	sigs=[]
	for bi in range(len(bins)):
		val = np.mean([mrdict[anal][seed][0][bi] for seed in range(nseed)])
		shuff = np.mean([[mrdict[anal][seed][k][bi] for k in mrdict[anal][seed].keys() if k!=0] for seed in range(nseed)],0)
		sigs.append(np.sum(abs(val)<[abs(s) for s in shuff])/len(shuff))
		axs[a].hist(shuff,color=colors[bi])	
		axs[a].axvline(x=val,color=colors[bi],ls='--',lw=5)
	axs[a].legend(['Young, p='+str(sigs[0]), 'Old, p='+str(sigs[1])])
	axs[a].set_title(anal)
fig.tight_layout()
fig.savefig(figurepath+'Motion/FramewiseDisplacement.png')
		
fig, ax = plt.subplots(figsize=(5, 5))
vals = []
for bi in range(len(bins)):
	vals.append(np.mean([mrdict['vals'][seed][0][bi] for seed in range(nseed)],0))
histbins=np.histogram(np.hstack((vals[0],vals[1])), bins=15)[1] #get the bin edges
for bi in range(len(bins)):
	ax.hist(vals[bi], histbins,color=colors[bi])
ax.legend(['Young', 'Old'])
fig.tight_layout()

fig.savefig(figurepath+'Motion/FramewiseDisplacement_.png')
			
			
		
	
						
				
		

					
	
	
	


			


for idx,vox in enumerate(mrvect):
	print(idx,vox)
	for k in phenol.keys():
		for task in tasks:
			if vox in mrmean[task][k]: # not all cosines in TP
				ts = {}
				for t in ['Motion','ISC']:
					m = dd.io.load(MRf,'/'+vox+'/'+k+'/'+task+'/mean') if t == 'Motion' else dd.io.load(MRf,'/'+vox+'/'+k+'/'+task+'/iscs')
					ts[t] = [np.mean((m['0'][0],m['0'][1]),axis=1),
							 np.std ((m['0'][0],m['0'][1]),axis=1),
					         1-np.sum([abs(np.mean(m['0'][0])-np.mean(m['0'][1]))
							          >abs(np.mean(m[str(s)][0])-np.mean(m[str(s)][1])) 
								 	  for s in np.arange(1,nshuff+1)])/nshuff]
				if ts[t][2] < 0.05:
						print(task,k,t)
						for h in [0,1]:
							print('Group',str(h),':',round(ts[t][0][h],2),'+/-',round(ts[t][1][h],2))
						print(t,'Difference p = ',ts[t][2])
						

# plot motion regressors to make sure they're correctly labeled
%matplotlib inline
import matplotlib.pyplot as plt
t = np.arange(1,len(Dtmp)+1)
fig, axs = plt.subplots(len(mrvect), 1)
for s in range(3):
	Dtmp = dd.io.load(subord[s],['/'+'DM'+'/reg'])[0]
	for i,mr in enumerate(mrvect):
		axs[i].plot(t,Dtmp[:,i])
		axs[i].set_title(mr)
fig.set_size_inches(10, 40)
fig.tight_layout()
fig.savefig(figurepath+'motion_regressors.png')


mrnc = [m for m in mrvect if 'Cosine' not in m]
fig, axs = plt.subplots(len(mrnc), 1)
for i,mr in enumerate(mrnc):
	axs[i].hist(mrmean[task]['age'][mr][1],bins=40,label='Young')
	axs[i].hist(mrmean[task]['age'][mr][0],bins=40,label='Old')
	#axs[i].hist(np.concatenate([mrmean[task]['all'][mr][1],mrmean[task]['all'][mr][0]]),bins=40)
	axs[i].set_title(mr)
axs[i].legend()
fig.set_size_inches(10, 30)
fig.tight_layout()
fig.savefig(figurepath+'motion_hist.png')


mrnc = [m for m in list(MR.keys()) if 'Cosine' not in m]
fig, axs = plt.subplots(len(mrnc), 1)
for i,mr in enumerate(mrnc):
	axs[i].hist(MR[mr]['age']['DM']['iscs']['0'][0][~np.isnan(MR[mr]['age']['DM']['iscs']['0'][0])],bins=40,label='Young')
	axs[i].hist(MR[mr]['age']['DM']['iscs']['0'][1][~np.isnan(MR[mr]['age']['DM']['iscs']['0'][1])],bins=40,label='Old')
	#axs[i].hist(np.concatenate([mrmean[task]['all'][mr][1],mrmean[task]['all'][mr][0]]),bins=40)
	axs[i].set_title(mr)
axs[i].legend()
fig.set_size_inches(10, 30)
fig.tight_layout()
fig.savefig(figurepath+'isc_motion_hist.png')

				
				
				