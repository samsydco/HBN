#!/usr/bin/env python3

# Add t-test (diff in mean,std) <- stats across subj
# Add ISC for each "vox"

import os
import glob
import h5py
import tqdm
import random
import numpy as np
import deepdish as dd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from ISC_settings import *

badsubs = ['sub-NDARNT043XGH.h5','sub-NDARUC851WHU.h5']

def ISCe_calc(isc_w):
	ISCe = isc_w[1]-isc_w[0]
	return ISCe
def ISCg_calc(isc_w,isc_b):
    ISCg = sum(isc_b)/4/(np.sqrt(isc_w[1])*np.sqrt(isc_w[0]))
    return ISCg

nsub = 41
bins = [0,4]
nTR=[750,250]

mrvect =['CSF','WhiteMatter','FramewiseDisplacement','Cosine00','Cosine01','Cosine02','Cosine03','Cosine04','Cosine05','Cosine06','Cosine07','X','Y','Z','RotX','RotY','RotZ','Xdiff','Ydiff','Zdiff','RotXdiff','RotYdiff','RotZdiff']
tasks = ['DM','TP']
anals = ['mean','isc_w']#'isc_b']#,'e_diff','g_diff']
mrdict = {t:{v:{a:{s:[] for s in range(nshuff+1)} for a in anals} for v in mrvect} for t in tasks}
					
for ti,task in enumerate(tasks):
	n_time = nTR[ti]
	print(task)
	mrvecttmp = mrvect if task == 'DM' else mrvect[:5]+mrvect[11:]
	mrlen = len(mrvecttmp)
	D = np.empty((nsub*2,mrlen,n_time),dtype='float16')
	Age = []
	Sex = []
	for bi,b in enumerate(bins):
		subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
		Sex.extend([Phenodf['Sex'][Phenodf['EID'] == shortsub(sub)].iloc[0] for sub in subl])
		Age.extend([bi]*len(subl))
		sub_ = 0 if b==0 else nsub # young and then old in D
		for sidx, sub in enumerate(subl):
			if task=='DM' or not any(s in sub for s in ['sub-NDARNT043XGH.h5','sub-NDARUC851WHU.h5']):
				D[sidx+sub_] = dd.io.load(sub,['/'+task+'/reg'])[0].T
			else:
				Dtmp = dd.io.load(sub,['/'+task+'/reg'])[0].T
				D[sidx+sub_,:4] = Dtmp[:4]
				D[sidx+sub_,5:] = Dtmp[4:]
	Ageperm = Age
	for shuff in tqdm.tqdm(range(nshuff+1)):
		subh = even_out(Age,Sex)
		group = np.zeros((2,2,nsub//2,mrlen,n_time))
		for h in [0,1]:
			for htmp in [0,1]:
				for i,sub in enumerate(subh[h][htmp]):
					group[h,htmp,i] = D[sub]
		for vi,v in enumerate(mrvecttmp):
			elimsub = np.where(np.isinf(group[:,:,:,vi]))[2]
			if len(elimsub)>0:
				elimsub=elimsub[0]
			for bi,b in enumerate(bins):
				mrdict[task][v]['isc_w'][shuff].append(pearsonr(\
					np.mean(np.delete(group[bi,0,:,vi],elimsub,0),axis=0),\
					np.mean(np.delete(group[bi,1,:,vi],elimsub,0),axis=0))[0])
				mrdict[task][v]['mean'][shuff].append(np.mean(np.concatenate((\
					np.delete(group[bi,0,:,vi],elimsub,0),\
					np.delete(group[bi,1,:,vi],elimsub,0)))))
			for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						#mrdict[task][v]['isc_b'][shuff].append(pearsonr(\
						#	np.mean(np.delete(group[0,htmp1,:,vi],elimsub,0),axis=0),\
						#	np.mean(np.delete(group[1,htmp2,:,vi],elimsub,0),axis=0))[0])
			#mrdict[task][v]['e_diff'][shuff] = ISCe_calc(mrdict[task][v]['isc_w'][shuff])
			#mrdict[task][v]['g_diff'][shuff] = ISCg_calc(mrdict[task][v]['isc_w'][shuff],\
			#											 mrdict[task][v]['isc_b'][shuff])
		# Now shuffle Age:
		random.shuffle(Age)
		
colors = ['#1f77b4', '#ff7f0e']
plt.rcParams.update({'font.size': 22})
for task in tasks:
	mrvecttmp = mrvect if task == 'DM' else mrvect[:5]+mrvect[11:]
	for v in mrvecttmp:
		fig, axs = plt.subplots(len(anals), 1,figsize=(10, 10))
		sigst = ''
		for a,anal in enumerate(['mean','isc_w']):
			sigs=[]
			for bi in range(len(bins)):
				val = mrdict[task][v][anal][0][bi]
				shuff = [mrdict[task][v][anal][k][bi] for k in mrdict[task][v][anal].keys() if k!=0]
				sigs.append(np.sum(abs(val)<[abs(s) for s in shuff])/len(shuff))
				axs[a].hist(shuff,color=colors[bi])	
				axs[a].axvline(x=val,color=colors[bi],ls='--',lw=5)
			axs[a].legend(['Young, p='+str(sigs[0]), 'Old, p='+str(sigs[1])])
			axs[a].set_title(anal)
			if any(s<0.05 for s in sigs):
				sigst='*'
		fig.tight_layout()
		fig.savefig(figurepath+'Motion/'+'_'.join([task,v,sigst])+'.png')
		

			
			
		
	
						
				
		

					
	
	
	


			


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


# after loading: MR=dd.io.load(glob.glob(ISCpath+'MR_*.h5')[0])
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

				
				
				