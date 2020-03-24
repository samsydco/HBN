#!/usr/bin/env python3

# searchlight HMM:
# look at young v old ll diff, and auc diff in searchlights across brain

import os
import glob
import tqdm
import deepdish as dd
from sklearn.model_selection import KFold
import brainiak.eventseg.event
from HMM_settings import *
SLlist = dd.io.load(ISCpath+'SLlist.h5')

savedir = HMMpath+'SL/'
nsub = 41
nvox = 81924//2
kf = KFold(n_splits=nsplit,shuffle=True)


for ti,task in enumerate(tasks):
	nTR_ = nTR[ti]
	for hem in ['L','R']:
		print(task,hem)
		subsavedir = savedir+task+'/'+hem+'/'
		SLs = SLlist[hem]
		SLdict  = {key: [] for key in ['best_k','auc_diff','ll_diff']}
		voxdict = {key: np.zeros(nvox) for key in ['best_k','auc_diff','ll_diff']}
		voxcount = np.zeros(nvox)
		if not os.path.exists(subsavedir):
			os.makedirs(subsavedir)
		else:
			SLs = SLs[len(glob.glob(subsavedir+'*')):]
			if len(glob.glob(subsavedir+'*')) > 0:
				maxSL = np.max([int(SL.split('_')[-1][:-3]) for SL in glob.glob(subsavedir+'*')])
				loaddict = dd.io.load(subsavedir+'_'.join([task,hem,str(maxSL)])+'.h5')
				SLdict  = loaddict['SLdict']
				voxdict = loaddict['voxdict']
				voxcount = loaddict['voxcount']
		D = np.empty((nsub*2,nvox,nTR_),dtype='float16')
		for b in bins:
			subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
			sub_ = 0 if b==0 else nsub # young and then old in D
			for sidx, sub in enumerate(subl):
				D[sidx+sub_] = dd.io.load(sub,['/'+task+'/'+hem])[0]
		for vi,voxl in tqdm.tqdm(enumerate(SLs)):
			Dsl = D[:,voxl,:]
			Dsplit = [D[:nsub],D[nsub:]] # split young and old
			tune_ll_both = np.zeros((nsplit,len(k_list)))
			for split in range(nsplit):
				LI,LO = next(kf.split(np.arange(nsub)))
				Dtrain = [np.mean(Dsplit[0][LI],axis=0).T,
						  np.mean(Dsplit[1][LI],axis=0).T]
				Dtest =  [np.mean(Dsplit[0][LO],axis=0).T,
						  np.mean(Dsplit[1][LO],axis=0).T]
				for ki,k in enumerate(k_list):
					hmm = brainiak.eventseg.event.EventSegment(n_events=k)
					hmm.fit(Dtrain)
					_, tune_ll_both[split,ki] = \
									hmm.find_events(np.mean(np.dstack(Dtest),axis=2))
			best_k = k_list[np.argmax(np.mean(tune_ll_both,axis=0))]
			voxdict['best_k'][voxl] += best_k
			voxcount['best_k'][voxl] += 1
			SLdict['best_k'].append(best_k)
			hmm = brainiak.eventseg.event.EventSegment(n_events=best_k)
			hmm.fit([np.mean(d,axis=0).T for d in Dsplit])
			auc = []
			for bi in range(len(bins)):
				auc.append(np.dot(hmm.segments_[bi], np.arange(best_k)).sum())
			auc_diff = (auc[1]-auc[0])/(best_k)*TR
			voxdict['auc_diff'][voxl] += (auc[1]-auc[0])/(best_k)*TR
			SLdict['auc_diff'].append(auc_diff)
			tune_ll_bin = np.zeros((2,nsplit))
			for split in range(nsplit):
				LI,LO = next(kf.split(np.arange(nsub)))
				Dtrain = [np.mean(Dsplit[0][LI],axis=0).T,
						  np.mean(Dsplit[1][LI],axis=0).T]
				Dtest =  [np.mean(Dsplit[0][LO],axis=0).T,
						  np.mean(Dsplit[1][LO],axis=0).T]
				hmm = brainiak.eventseg.event.EventSegment(n_events=best_k)
				hmm.fit(Dtrain)
				for bi,b in enumerate(bins):
					_, tune_ll_bin[bi,nsplit] = \
							hmm.find_events(Dtest[bi]) # tune_ll per age group
				ll_diff = np.mean(tune_ll_bin[1]) - np.mean(tune_ll_bin[0])
				voxdict['ll_diff'][voxl].append(ll_diff)
				SLdict['ll_diff']
			voxcount[voxl] += 1
			dd.io.save(subsavedir+'_'.join([task,hem,str(vi)])+'.h5',\
					   {'voxdict':voxdict, 'SLdict':SLdict, 'voxcount':voxcount})
		for k,v in voxdict:
			voxdict[k] = v / voxcount
		dd.io.save(savedir+'_'.join([task,hem])+'.h5', \
				   {'voxdict':voxdict, 'SLdict':SLdict, 'voxcount':voxcount})
			
			
	
