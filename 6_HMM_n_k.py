#!/usr/bin/env python3

# Determine if k at max LL is similar in 
# Youngest group (bin 0) and 
# Oldest group (bin 4)
# Is the assumption true that the number of events is equal across groups?

import os
import tqdm
import brainiak.eventseg.event
from HMM_settings import *

bins = [0,4]
nbins = len(bins)
nshuff=100
nshuff2perm=1000
task = 'DM'

if os.path.exists(nkh5):
	pdict=dd.io.load(nkh5)

for roi in tqdm.tqdm(glob.glob(roidir+'*h5')):
	roi_short = roi.split('/')[-1][:-3]
	if (os.path.exists(nkh5) and pdict[roi_short]['k_diff_p'] <= 0.05) or \
	not os.path.exists(nkh5):
		if os.path.exists(nkh5):
			roidict = dd.io.load(nkdir+roi+'.h5')
			nshuff_ = len(roidict['0']['best_k'])-1
			Dall = [roidict['0']['D'],roidict['4']['D']]
			nshuff2 = nshuff2perm + nshuff_
			shuffl = np.arange(nshuff_+1,nshuff2+1)
		else:
			roidict = {str(b):{} for b in bins}
			Ddict = dd.io.load(roi)
			for b in bins:
				roidict[str(b)]['D'] = Ddict[task]['bin_'+str(b)]['D']
			Dall = [roidict['0']['D'],roidict['4']['D']]
			shuffl = np.arange(nshuff+1)
		for bi,b in enumerate(bins):
			notbi = 1 if bi==0 else 0
			best_k = np.zeros(len(shuffl))
			tune_ll = np.zeros((len(shuffl),nsplit,len(k_list)))
			for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
				Dtrain = np.mean(Dall[bi][Ls[0]],axis=0).T
				Dtest_all  = np.concatenate((Dall[bi][Ls[1]],Dall[notbi][Ls[1]]),axis=0)
				subl = np.concatenate((np.ones(len(Ls[1])),np.zeros(len(Ls[1]))),axis=0)
				for ki,k in enumerate(k_list):
					hmm = brainiak.eventseg.event.EventSegment(n_events=k)
					hmm.fit(Dtrain)
					for shuff in range(len(shuffl)):
						if shuff != 0: # RANDOMIZE
							subl = np.random.permutation(subl)
						_, tune_ll[shuff,split,ki] = hmm.find_events(np.mean(Dtest_all[subl==1], axis=0).T)			
			for shuff in range(len(shuffl)):
				best_k[shuff] = np.mean([k_list[np.argmax(tune_ll[shuff,ki])] for ki in range(kf.n_splits)])
			if len(roidict[str(b)].keys())>0:
				roidict[str(b)]['best_k'] = np.append(data[str(b)]['best_k'],best_k,axis=0)
				roidict[str(b)]['tune_ll'] = np.append(data[str(b)]['tune_ll'], tune_ll,axis=0)
			else:
				roidict[str(b)]['tune_ll'] = tune_ll
				roidict[str(b)]['best_k'] = best_k
		roidict['k_diff'] = roidict[str(4)]['best_k'] - roidict[str(0)]['best_k']
		dd.io.save(nkdir+roi_short+'.h5',roidict)
	

roidict = {}
for roi in tqdm.tqdm(glob.glob(nkdir+'*.h5')):
	roi_short = roi.split('/')[-1][:-3]
	roidict[roi_short] = {b:[] for b in [str(b) for b in bins]+['k_diff_p','sig']}
	data = dd.io.load(roi)
	for b in bins:
		roidict[roi_short][str(b)] = data[str(b)]['best_k'][0]
	roidict[roi_short]['k_diff_p'] = np.sum(abs(data['k_diff'][0])<abs(data['k_diff'][1:]))/nshuff
	roidict[roi_short]['sig'] = 1 if roidict[roi_short]['k_diff_p']<0.05 else 0
		
dd.io.save(nkh5,roidict)
	
