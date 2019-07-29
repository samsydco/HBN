#!/usr/bin/env python3

# Add t-test (diff in mean,std) <- stats across subj
# Add ISC for each "vox"

import os
import glob
import h5py
import tqdm
import numpy as np
import deepdish as dd
from datetime import date
from scipy.spatial.distance import squareform
from scipy.stats import ttest_ind
from settings import *
from ISC_settings import *
phenolperm = phenol

if len(glob.glob(ISCpath+'MR_*.h5'))>1:
	MR=dd.io.load(glob.glob(ISCpath+'MR_*.h5')[0])
else:
	
	
MRf = ISCpath+'MR_'+str(date.today())+'.h5'
if os.path.exists(MRf):
    os.remove(MRf)
mrvect =['CSF','WhiteMatter','FramewiseDisplacement','Cosine00','Cosine01','Cosine02','Cosine03','Cosine04','Cosine05','Cosine06','Cosine07','X','Y','Z','RotX','RotY','RotZ','Xdiff','Ydiff','Zdiff','RotXdiff','RotYdiff','RotZdiff']
tasks = ['DM','TP']
with h5py.File(MRf) as hf:
	for vox in mrvect:
		grp = hf.create_group(vox)
		for k in phenol.keys():
			grpk = grp.create_group(k)
			for task in tasks:
				grpt = grpk.create_group(task)
				for anal in ['mean','iscs']:
					grpa = grpt.create_group(anal)

#mrmean = dict(zip(tasks,(dict(zip(phenol.keys(),({} for k in phenol.keys()))) for task in tasks)))
#mriscs = dict(zip(tasks,(dict(zip(phenol.keys(),({} for k in phenol.keys()))) for task in tasks)))
nshuff = 100
subsh = []
for task in tasks:
	print(task)
	mrvecttmp = mrvect if task == 'DM' else mrvect[:5]+mrvect[11:]
	sh = dd.io.load(subord[0],['/'+task+'/reg'])[0].shape
	D = np.empty(((len(subord),)+sh))
	for sidx, sub in enumerate(subord):
		Dtmp = dd.io.load(sub,['/'+task+'/reg'])[0]
		if Dtmp.shape[1] == sh[1]:
			D[sidx,:,:] = Dtmp
		elif Dtmp.shape[1]+1 == sh[1]:
			if not any(s in sub for s in ['sub-NDARNT043XGH.h5','sub-NDARUC851WHU.h5']):
				subsh.append(sub)
			D[sidx,:,:] = np.concatenate((Dtmp[:,:4],Dtmp[:,:1]*np.nan,Dtmp[:,4:]),axis=1)
	D = np.transpose(D,(2,1,0))
	n_vox,n_time,n_subj=D.shape
	#for k,v in phenolperm.items():
	#	mrmean[task][k] = dict(zip(mrvecttmp,[[[],[]] for i in mrvecttmp]))
	#	mriscs[task][k] = dict(zip(mrvecttmp,[[[],[]] for i in mrvecttmp]))
	for shuff in tqdm.tqdm(range(nshuff+1)):
		for k,v in phenolperm.items():
			v2 = phenolperm['sex'] if k!='sex' else phenolperm['age']
			subh = even_out(v,v2)
			g = [subh[0][1]+subh[0][0],subh[1][1]+subh[1][0]]
			for vox in range(n_vox):
				with h5py.File(MRf,'a') as hf:
					hf.create_dataset(mrvecttmp[vox]+'/'+k+'/'+task+'/mean/'+str(shuff),\
						data=[np.mean(D[vox,:,g[h]],axis=1) for h in [0,1]])
					hf.create_dataset(mrvecttmp[vox]+'/'+k+'/'+task+'/iscs/'+str(shuff),\
						data=[squareform(np.corrcoef(D[vox,:,g[h]].T), checks=False) for h in [0,1]])
				#for h in [0,1]:
					#mrmean[task][k][mrvecttmp[vox]][h].append(np.mean(D[vox,:,g[h]],axis=1))
					#mriscs[task][k][mrvecttmp[vox]][h].append(squareform(np.corrcoef(D[vox,:,g[h]].T), checks=False))
		# randomly shuffle phenol:
		for k,v in phenol.items():
			nonnanidx = np.argwhere(~np.isnan(phenol[k]))
			randidx = np.random.permutation(nonnanidx)
			phenol[k] = [v[randidx[nonnanidx==idx][0]] if idx in nonnanidx else i for idx,i in enumerate(v)]

			


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

				
				
				