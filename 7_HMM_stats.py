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

lldict = dd.io.load(llh5)
kdict=dd.io.load(nkh5)
df = pd.DataFrame(kdict).T.merge(pd.DataFrame(lldict).T, left_index=True, right_index=True, how='inner')
df=df[((df['0_2k_diff']>ll_thresh) | (df['4_2k_diff']>ll_thresh))]
ROIl = list(df.index)

savedir = HMMpath+'shuff_5bins_train04_paper/'
bins = np.arange(nbinseq)
nbins = len(bins)
task = 'DM'
nTR_ = nTR[0]
nshuff2=1000

for roi_short in ROIl:
	roif = roidir+roi_short+'.h5'
	if os.path.exists(savedir+roi_short+'.h5'):
		roidict = dd.io.load(savedir+roi_short+'.h5')
	else:
		roidict = {t:{'bin_'+str(b):{} for b in bins} for t in ['DM']}
		
	
	for b in bins:
		roidict[task]['bin_'+str(b)]['D'] =  dd.io.load(roif, task+'/bin_'+str(b)+'/D')
	D = [roidict[task]['bin_'+str(b)]['D'] for b in bins]
	bin_tmp = [0,4]
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
	roidict[task]['best_k'] = best_k
	tune_seg = np.zeros((nshuff+1,nbins,nsplit,nTR_,best_k))
	tune_ll = np.zeros((nshuff+1,nbins,nsplit))
	for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
		Dtrain = [np.mean(d[Ls[0]],axis=0).T for d in [D[bi] for bi in bin_tmp]]
		Dtest_all  = np.concatenate([d[Ls[1]] for d in D])
		nsubLO = len(Ls[1])
		subl = np.arange(nsubLO*nbins) # subject list to be permuted!
		hmm = brainiak.eventseg.event.EventSegment(n_events=best_k)
		hmm.fit(Dtrain)
		for shuff in range(nshuff+1):		
			for bi in range(nbins):
				idx = np.arange(bi*nsubLO,(bi+1)*nsubLO)
				Dtest = np.mean(Dtest_all[subl[idx]],axis=0).T
				tune_seg[shuff,bi,split], tune_ll[shuff,bi,split] = hmm.find_events(Dtest)
			# RANDOMIZE
			subl = np.random.permutation(nsubLO*nbins)
	roidict[task]['tune_ll_perm'] = tune_ll
	roidict[task]['tune_seg_perm'] = tune_seg
		
		
	
	

				for shuff in range(nshuff+1):
					shuffstr = 'shuff_'+str(shuff)
					roidict[task][shuffstr] = {}
					E_k = []
					auc = []
					for bi in range(nbins):
						E_k.append(np.dot(np.mean(tune_seg[shuff,bi],axis=0), np.arange(best_k)+1))
						auc.append(E_k[bi].sum())
					roidict[task][shuffstr]['E_k'] = E_k
					roidict[task][shuffstr]['auc'] = auc
					roidict[task][shuffstr]['auc_diff'] = ((auc[-1]-auc[0])/best_k)*TR
					roidict[task][shuffstr]['ll_diff'] = np.mean(np.diff(np.mean(tune_ll[shuff],axis=1)))/nTR_
			dd.io.save(savedir+roi_short+'.h5',roidict)
				
			
		
		
	
	