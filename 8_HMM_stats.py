#!/usr/bin/env python3

'''
# In ROIs with similar k's and high LL's in both or one age group:

1) Get patterns by HMM([Young_train, Old_train])
2) LLs = find_events(Young_test), find_events(Old_test)
3) Pick K by maximizing avg LLs
4A) LL diff at this K
4B) AUC diff at this K
Permutations only for 4
Perm: find_events(Perm1), find_events(Perm2)
All of this train/test set loop
'''

import os
import glob
import tqdm
import numpy as np
import deepdish as dd
import brainiak.eventseg.event
from HMM_settings import *

savedir = HMMpath+'shuff_5bins_train04_'
bins = np.arange(nbinseq)
nbins = len(bins)
bin_tmp = [0,4]
task = 'DM'
nTR_ = nTR[0]
nshuff2perm=1000


for seed in tqdm.tqdm(seeds):
	seedsavedir = savedir+seed
	if not os.path.exists(seedsavedir): os.makedirs(seedsavedir)
	for roi_short in tqdm.tqdm(ROIl):
		roif = roidir+seed+'/'+roi_short+'.h5'
		if os.path.exists(seedsavedir+'/'+roi_short+'.h5'):
			roidict = dd.io.load(seedsavedir+'/'+roi_short+'.h5')
			nshuff_ = len(roidict['ll_diff'][1:])
			p_ll_ = np.sum(abs(roidict['ll_diff'][0])<abs(roidict['ll_diff'][1:]) )/nshuff_
			p_auc = np.sum(abs(roidict['auc_diff'][0])<abs(roidict['auc_diff'][1:]) )/nshuff_
			nshuff2 = nshuff2perm + nshuff_
			shuffl = np.arange(nshuff_+1,nshuff2+1)
			nshuff_all = 0
			p_ll_all = p_aucall = 1
			if os.path.exists(ISCpath+'p_vals_seeds.h5'):
				nshuff_all = len(dd.io.load(ISCpath+'p_vals_seeds.h5', '/roidict/'+roi_short+'/auc_diff/shuff'))
				p_ll_all = dd.io.load(ISCpath+'p_vals_seeds.h5', '/roidict/'+roi_short+'/ll_diff/p')
				p_aucall = dd.io.load(ISCpath+'p_vals_seeds.h5', '/roidict/'+roi_short+'/auc_diff/p')
		else:
			roidict = {'bin_'+str(b):{} for b in bins}
			for b in bins:
				roidict['bin_'+str(b)]['D'] =  dd.io.load(roif, '/DM/bin_'+str(b)+'/D')
			nshuff2 = nshuff_ = nshuff
			shuffl = np.arange(nshuff2+1)
			p_ll_ = p_auc = 0 #Default = Do the test
		#if ((p_ll_<0.05 or p_auc<0.05) and nshuff_ < nshuff2perm) \
		#or (p_ll_== 0 or p_auc == 0) or (nshuff_ < nshuff_all and (p_ll_all < 0.05 or p_aucall < 0.05)):
		if (nshuff_ < nshuff_all and (p_ll_all < 0.05 or p_aucall < 0.05)):
			D = [roidict['bin_'+str(b)]['D'] for b in bins]
			if not os.path.exists(seedsavedir+'/'+roi_short+'.h5'):
				tune_ll = np.zeros((nbins,nsplit,len(k_list)))
				for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
					Dtrain = [np.mean(d[Ls[0]],axis=0).T for d in [D[bi] for bi in bin_tmp]]
					Dtest  = [np.mean(d[Ls[1]],axis=0).T for d in D]
					for ki,k in enumerate(k_list):
						hmm = brainiak.eventseg.event.EventSegment(n_events=k)
						hmm.fit(Dtrain)
						for b in bins:
							_, tune_ll[b,split,ki] = hmm.find_events(Dtest[b])
				#Now calculating best_k from average ll:
				best_k = k_list[np.argmax(np.mean(np.mean(tune_ll,0),0))]
				roidict['best_k'] = best_k
				tune_seg = np.zeros((len(shuffl),nbins,nsplit,nTR_,best_k))
				tune_ll = np.zeros((len(shuffl),nbins,nsplit))
			else:
				best_k = roidict['best_k']
				tune_seg = np.append(roidict['tune_seg_perm'], np.zeros((len(shuffl),nbins,nsplit,nTR_,best_k)),axis=0)
				tune_ll = np.append(roidict['tune_ll_perm'], np.zeros((len(shuffl),nbins,nsplit)),axis=0)
			for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
				Dtrain = [np.mean(d[Ls[0]],axis=0).T for d in [D[bi] for bi in bin_tmp]]
				Dtest_all  = np.concatenate([d[Ls[1]] for d in D])
				nsubLO = len(Ls[1])
				subl = np.arange(nsubLO*nbins) # subject list to be permuted!
				hmm = brainiak.eventseg.event.EventSegment(n_events=best_k)
				hmm.fit(Dtrain)
				for shuff in shuffl:		
					for bi in range(nbins):
						if shuff != 0:
							# RANDOMIZE
							subl = np.random.permutation(nsubLO*nbins)
						idx = np.arange(bi*nsubLO,(bi+1)*nsubLO)
						Dtest = np.mean(Dtest_all[subl[idx]],axis=0).T
						tune_seg[shuff,bi,split], tune_ll[shuff,bi,split] = hmm.find_events(Dtest)
			roidict['tune_ll_perm'] = tune_ll
			roidict['tune_seg_perm'] = tune_seg
	
			roidict['E_k'] = np.zeros((nshuff2+1,nbins,nTR_))
			roidict['auc'] = np.zeros((nshuff2+1,nbins))
			roidict['auc_diff'] = np.zeros(nshuff2+1)
			roidict['ll_diff'] = np.zeros(nshuff2+1)
			for shuff in range(nshuff2+1):
				for bi in range(nbins):
					roidict['E_k'][shuff,bi] = np.dot(np.mean(tune_seg[shuff,bi],axis=0), np.arange(best_k)+1)
					roidict['auc'][shuff,bi] = roidict['E_k'][shuff,bi].sum()
				roidict['auc_diff'][shuff] = ((roidict['auc'][shuff,-1] -roidict['auc'][shuff,0])/best_k)*TR
				roidict['ll_diff'][shuff] = np.mean(tune_ll[shuff,-1] - tune_ll[shuff,0])/nTR_
			dd.io.save(seedsavedir+'/'+roi_short+'.h5',roidict)
				
			
		
		
	
	