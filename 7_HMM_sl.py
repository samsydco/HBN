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
y = [0]*int(np.floor(nsub/nsplit))*4+[1]*(int(np.floor(nsub/nsplit))+1)
kf = KFold(n_splits=nsplit, shuffle=True, random_state=2)


for ti,task in enumerate(tasks):
	nTR_ = nTR[ti]
	for hem in ['L','R']:
		print(task,hem)
		subsavedir = savedir+task+'/'+hem+'/'
		SLs = SLlist[hem]
		SLs = {key: SLs[key] for key in np.arange(len(SLs))}
		etcdict = {key: {} for key in np.arange(len(SLs))}
		SLdict  = {key: {key:[] for key in np.arange(nshuff+1)} \
				   for key in ['best_k','auc_diff','ll_diff']}
		voxdict = {key: np.zeros((nshuff+1,nvox)) for key in ['best_k','auc_diff','ll_diff']}
		voxcount = np.zeros(nvox)
		if not os.path.exists(subsavedir):
			os.makedirs(subsavedir)
		elif len(glob.glob(subsavedir+'*')) > 0:
			maxSL = np.max([int(SL.split('_')[-1][:-3]) for SL in glob.glob(subsavedir+'*')])
			SLs = dict((k,SLs[k]) for k in SLs.keys() if k > maxSL)
			loaddict = dd.io.load(subsavedir+'_'.join([task,hem,str(maxSL)])+'.h5')
			etcdict = loaddict['etcdict']
			SLdict  = loaddict['SLdict']
			voxdict = loaddict['voxdict']
			voxcount = loaddict['voxcount']
		D = np.empty((nsub*2,nvox,nTR_),dtype='float16')
		for b in bins:
			subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
			sub_ = 0 if b==0 else nsub # young and then old in D
			for sidx, sub in enumerate(subl):
				D[sidx+sub_] = dd.io.load(sub,['/'+task+'/'+hem])[0]
		for vi,voxl in tqdm.tqdm(SLs.items()):
			badvox = np.unique(np.where(np.isnan(D[:,voxl,:]))[1])
			voxl_tmp = np.array([v for i,v in enumerate(voxl) if not any(b==i for b in badvox)])
			etcdict[vi]['voxl'] = voxl_tmp
			voxcount[voxl_tmp] += 1
			Dsl = D[:,voxl_tmp,:]
			Dsplit = [Dsl[:nsub],Dsl[nsub:]] # split young and old
			tune_ll = np.zeros((2,nsplit,len(k_list)))
			tune_seg = {key:{key:np.zeros((nsplit,nTR_,key)) for key in k_list} for key in range(len(bins))}
			for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
				Dtrain = [np.mean(Dsplit[0][Ls[0]],axis=0).T,
						  np.mean(Dsplit[1][Ls[0]],axis=0).T]
				Dtest =  [np.mean(Dsplit[0][Ls[1]],axis=0).T,
						  np.mean(Dsplit[1][Ls[1]],axis=0).T]
				for ki,k in enumerate(k_list):
					hmm = brainiak.eventseg.event.EventSegment(n_events=k)
					hmm.fit(Dtrain)
					for bi in range(len(bins)):
						tune_seg[bi][k][split], tune_ll[bi,split,ki] = \
									hmm.find_events(Dtest[bi])
			etcdict[vi]['tune_ll'] = tune_ll
			etcdict[vi]['tune_seg'] = tune_seg
			best_k = k_list[np.argmax(np.mean(np.mean(tune_ll,axis=0),axis=0))]
			voxdict['best_k'][0,voxl_tmp] += best_k
			SLdict['best_k'][0].append(best_k)
			
			tune_seg = np.zeros((nshuff+1,2,nsplit,nTR_,best_k))
			tune_ll = np.zeros((nshuff+1,2,nsplit))
			for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
				Dtrain = [np.mean(Dsplit[0][Ls[0]],axis=0).T,
						  np.mean(Dsplit[1][Ls[0]],axis=0).T]
				Dtest_all =  np.concatenate([Dsplit[0][Ls[1]],Dsplit[1][Ls[1]]])
				nsubLO = len(Ls[1])
				subl = np.arange(nsubLO*2) # subject list to be permuted!
				hmm = brainiak.eventseg.event.EventSegment(n_events=best_k)
				hmm.fit(Dtrain)
				for shuff in range(nshuff+1):
					Dtest = [np.mean(Dtest_all[subl[:nsubLO]],axis=0).T,
							 np.mean(Dtest_all[subl[nsubLO:]],axis=0).T]
					for bi in range(len(bins)):
						tune_seg[shuff,bi,split], tune_ll[shuff,bi,split] = hmm.find_events(Dtest[bi])
					# RANDOMIZE
					subl = np.random.permutation(nsubLO*2)
			etcdict[vi]['tune_ll_perm'] = tune_ll
			etcdict[vi]['tune_seg_perm'] = tune_seg
			for shuff in range(nshuff+1):
				shuffstr = 'shuff_'+str(shuff)
				etcdict[vi][shuffstr] = {}
				E_k = []
				auc = []
				for bi in range(len(bins)):
					E_k.append(np.dot(np.mean(tune_seg[shuff,bi],axis=0), np.arange(best_k)+1))
					auc.append(E_k[bi].sum())
				etcdict[vi][shuffstr]['E_k'] = E_k
				etcdict[vi][shuffstr]['auc'] = auc
				auc_diff = ((auc[1]-auc[0])/best_k)*TR
				SLdict['auc_diff'][shuff].append(auc_diff)
				voxdict['auc_diff'][shuff,voxl_tmp] += auc_diff
				ll_diff = (np.mean(tune_ll[shuff,1,:]) - np.mean(tune_ll[shuff,0,:]))/nTR_
				SLdict['ll_diff'][shuff].append(ll_diff)
				voxdict['ll_diff'][shuff,voxl_tmp] += ll_diff
			dd.io.save(subsavedir+'_'.join([task,hem,str(vi)])+'.h5',\
					   {'voxdict':voxdict, 'SLdict':SLdict, 'voxcount':voxcount,'etcdict':etcdict})
		for k,v in voxdict.items():
			voxdict[k] = v / voxcount
		dd.io.save(savedir+'_'.join([task,hem])+'.h5', \
				   {'voxdict':voxdict, 'SLdict':SLdict, 'voxcount':voxcount, 'etcdict':etcdict})
			
			
	

