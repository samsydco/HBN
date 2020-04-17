#!/usr/bin/env python3

# Use ll values determined in seperate code
# Permute subject IDs between young and old
# At k value with best ll compute:
# (a) ll difference between "young" and "old"
# (b) auc difference between "young" and "old"
# determine how fast 100 shuffs takes to determine if can run more,
# or need to do normal distribution approximation...

'''
New Process:

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
from sklearn.model_selection import KFold
import nibabel.freesurfer.io as free
from HMM_settings import *

ROInow = ROIopts[1]
HMMf = HMMpath+'timing_'+ROInow+'/'
savedir = HMMpath+'shuff/'
ROIs = glob.glob(HMMf+'*h5')
nsub= 41
y = [0]*int(np.floor(nsub/nsplit))*4+[1]*(int(np.floor(nsub/nsplit))+1)
kf = KFold(n_splits=nsplit, shuffle=True, random_state=2)


for hemi in glob.glob(path+'ROIs/annot/*'):
	print(hemi)
	lab = free.read_annot(hemi)
	h = hemi.split('/')[-1][0].capitalize()
	for ri,roi_tmp in enumerate(lab[2]):
		roi_short=roi_tmp.decode("utf-8")[11:]
		roidict = {}
		vall = np.where(lab[0]==ri)[0]
		for ti,task in enumerate(tasks):
			roidict[task] = {}
			nTR_ = nTR[ti]
			# Need to fit HMM for all k to find the best:
			for b in bins:
				if len(vall) > 0:
					roidict[task]['bin_'+str(b)] = {}
					subl = [ageeq[i][1][b][idx] for i in [0,1] for idx in np.random.choice(lenageeq[i][b],minageeq[i],replace=False)]
					roidict[task]['bin_'+str(b)]['subl'] = subl
					nsub = len(subl)
					# Load data
					D = np.empty((nsub,len(vall),nTR_),dtype='float16')
					badvox = []
					for sidx, sub in enumerate(subl):
						D[sidx,:,:] = dd.io.load(sub,['/'+task+'/'+h], sel=dd.aslice[vall,:])[0]
						badvox.extend(np.where(np.isnan(D[sidx,:,0]))[0]) # Some subjects missing some voxels
					D = np.delete(D,badvox,1)
					vall = np.delete(vall,badvox)
					roidict['vall'] = vall
					roidict[task]['bin_'+str(b)]['D'] = D
			if len(vall) > 0:
				D = [roidict[task]['bin_0']['D'],roidict[task]['bin_4']['D']]
				tune_ll = np.zeros((2,nsplit,len(k_list)))
				tune_seg = {key:{key:np.zeros((nsplit,nTR_,key)) for key in k_list} for key in range(len(bins))}
				for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
					Dtrain = [np.mean(D[0][Ls[0]],axis=0).T,
							  np.mean(D[1][Ls[0]],axis=0).T]
					Dtest =  [np.mean(D[0][Ls[1]],axis=0).T,
							  np.mean(D[1][Ls[1]],axis=0).T]
					for ki,k in enumerate(k_list):
						hmm = brainiak.eventseg.event.EventSegment(n_events=k)
						hmm.fit(Dtrain)
						for bi in range(len(bins)):
							tune_seg[bi][k][split], tune_ll[bi,split,ki] = \
										hmm.find_events(Dtest[bi])
				roidict[task]['tune_ll'] = tune_ll
				roidict[task]['tune_seg'] = tune_seg
				#Now calculating best_k from average ll:
				best_k = k_list[np.argmax(np.mean(np.mean(tune_ll,0),0))]
				roidict[task]['best_k'] = best_k
				tune_seg = np.zeros((nshuff+1,2,nsplit,nTR_,best_k))
				tune_ll = np.zeros((nshuff+1,2,nsplit))
				for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
					Dtrain = [np.mean(D[0][Ls[0]],axis=0).T,
							  np.mean(D[1][Ls[0]],axis=0).T]
					Dtest_all =  np.concatenate([D[0][Ls[1]],D[1][Ls[1]]])
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
				roidict[task]['tune_ll_perm'] = tune_ll
				roidict[task]['tune_seg_perm'] = tune_seg
				for shuff in range(nshuff+1):
					shuffstr = 'shuff_'+str(shuff)
					roidict[task][shuffstr] = {}
					E_k = []
					auc = []
					for bi in range(len(bins)):
						E_k.append(np.dot(np.mean(tune_seg[shuff,bi],axis=0), np.arange(best_k)+1))
						auc.append(E_k[bi].sum())
					roidict[task][shuffstr]['E_k'] = E_k
					roidict[task][shuffstr]['auc'] = auc
					roidict[task][shuffstr]['auc_diff'] = ((auc[1]-auc[0])/best_k)*TR
					roidict[task][shuffstr]['ll_diff'] = (np.mean(tune_ll[shuff,1,:]) - np.mean(tune_ll[shuff,0,:]))/nTR_
		if len(vall) > 0:
			dd.io.save(savedir+roi_short+'.h5',roidict)
				
			
				
        
        

'''

for roi in tqdm.tqdm(ROIs):
	roi_short = roi.split('/')[-1][:-3]
	ROIsHMM = dd.io.load(roi)
	roidict = {key: {} for key in tasks}
	if os.path.exists(savedir+roi_short+'.h5'):
		roidict=dd.io.load(savedir+roi_short+'.h5')
		for ti,task in enumerate(tasks):
			roidict[task]['best_k'] = k_list[np.argmax(np.mean(np.concatenate((ROIsHMM[task]['bin_0']['tune_ll'],ROIsHMM[task]['bin_4']['tune_ll'])),0))]
	else:
		for ti,task in enumerate(tasks):
			nTR_ = nTR[ti]
			#Now calculating best_k from average ll:
			best_k = k_list[np.argmax(np.mean(np.concatenate((ROIsHMM[task]['bin_0']['tune_ll'],ROIsHMM[task]['bin_4']['tune_ll'])),0))]
			roidict[task]['best_k'] = best_k
			D = [ROIsHMM[task]['bin_0']['D'],ROIsHMM[task]['bin_4']['D']]
			tune_seg = np.zeros((nshuff+1,2,nsplit,nTR_,best_k))
			tune_ll = np.zeros((nshuff+1,2,nsplit))
			for split in range(nsplit):
				LI,LO = next(kf.split(np.arange(nsub)))
				Dtrain = [np.mean(D[0][LI],axis=0).T,
						  np.mean(D[1][LI],axis=0).T]
				Dtest_all =  np.concatenate([D[0][LO],D[1][LO]])
				nsubLO = len(LO)
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
			roidict[task]['tune_ll'] = tune_ll
            for shuff in range(nshuff+1):
				shuffstr = 'shuff_'+str(shuff)
				roidict[task][shuffstr] = {}
                E_k = []
				auc = []
				for bi in range(len(bins)):
                    E_k.append(np.dot(np.mean(tune_seg[shuff,bi],axis=0), np.arange(best_k)+1))
					auc.append(E_k[bi].sum())
                roidict[task][shuffstr]['E_k'] = E_k
                roidict[task][shuffstr]['E_k'] = auc
				roidict[task][shuffstr]['auc_diff'] = ((auc[1]-auc[0])/best_k)*TR
				roidict[task][shuffstr]['ll_diff'] = np.mean(tune_ll[shuff,1,:]) - np.mean(tune_ll[shuff,0,:])
	dd.io.save(savedir+roi_short+'.h5',roidict)
		

'''
		
		
	
	