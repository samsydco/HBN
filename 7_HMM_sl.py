#!/usr/bin/env python3

# searchlight HMM:
# look at young v old ll diff, and auc diff in searchlights across brain

import tqdm
import deepdish as dd
from sklearn.model_selection import KFold
from HMM_settings import *
SLlist = dd.io.load(ISCpath+'SLlist.h5')

savedir = HMMpath+'SL/'
nsub = 41
nvox = 81924//2
kf = KFold(n_splits=nsplit,shuffle=True)


for ti,task in enumerate(tasks):
	nTR_ = nTR[ti]
	for hem in tqdm.tqdm(['L','R']):
		D = np.empty((nsub*2,nvox,nTR_),dtype='float16')
		for b in bins:
			subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
			sub_ = 0 if b==0 else nsub # young and then old in D
			for sidx, sub in enumerate(subl):
				D[sidx+sub_] = dd.io.load(sub,['/'+task+'/'+hem])[0]
		SLdict  = {key: [] for key in ['best_k','auc_diff','ll_diff']}
		voxdict = {key: np.zeros(n_vox) for key in ['best_k','auc_diff','ll_diff']}
		voxcount = np.zeros(n_vox)
		for sl in SLlist[hem]:
			voxl = SLlist[hem][sl]
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
			best_k = k_list[np.argmax(np.mean(tune_ll,axis=0))]
			voxdict['best_k'][voxl] += best_k
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
		for k,v in voxdict:
			voxdict[k] = v / voxcount
		dd.io.save(savedir+'_'.join([task,hem])+'.h5',{'voxdict':voxdict,'SLdict':SLdict})
			
			
	

