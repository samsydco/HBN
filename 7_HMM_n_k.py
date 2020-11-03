#!/usr/bin/env python3

# In parcels where there may be a n_k difference
# Run 1000 shuffles

import tqdm
import brainiak.eventseg.event
from sklearn.model_selection import KFold
from HMM_settings import *

sametimedir = HMMpath+'shuff_5bins_train04/'
nkdir = HMMpath+'nk_moreshuff/'#'nk/'
nsub= 41
y = [0]*int(np.floor(nsub/nsplit))*4+[1]*(int(np.floor(nsub/nsplit))+1)
kf = KFold(n_splits=nsplit, shuffle=True, random_state=2)
bins = [0,4]
nbins = len(bins)
nshuff2=1000
task = 'DM'

roidict=dd.io.load(HMMpath+'nk2.h5')

for roi in tqdm.tqdm(roidict.keys()):
	if roidict[roi]['k_diff_p'] <= 0.05:
		data = dd.io.load(nkdir+roi+'.h5')
		nshuff = len(data['0']['best_k'])-1
		Dall = [data['0']['D'],data['4']['D']]
		for bi,b in enumerate(bins):
			notbi = 1 if bi==0 else 0
			best_k = np.append(data[str(b)]['best_k'],np.zeros((nshuff2-nshuff)),axis=0)
			tune_ll = np.append(data[str(b)]['tune_ll'],np.zeros((nshuff2-nshuff,nsplit,len(k_list))),axis=0)
			for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
				Dtrain = np.mean(Dall[bi][Ls[0]],axis=0).T
				Dtest_all  = np.concatenate((Dall[bi][Ls[1]],Dall[notbi][Ls[1]]),axis=0)
				subl = np.concatenate((np.ones(len(Ls[1])),np.zeros(len(Ls[1]))),axis=0)
				for ki,k in enumerate(k_list):
					hmm = brainiak.eventseg.event.EventSegment(n_events=k)
					hmm.fit(Dtrain)
					for shuff in np.arange(nshuff+1,nshuff2+1):
						# RANDOMIZE
						subl = np.random.permutation(subl)
						_, tune_ll[shuff,split,ki] = hmm.find_events(np.mean(Dtest_all[subl==1], axis=0).T)
			for shuff in np.arange(nshuff+1,nshuff2+1):
				 best_k[shuff] = np.mean([k_list[np.argmax(tune_ll[shuff,ki])] for ki in range(kf.n_splits)])
			data[str(b)]['tune_ll'] = tune_ll
			data[str(b)]['best_k'] = best_k
		data['k_diff'] = data[str(4)]['best_k'] - data[str(0)]['best_k']
		dd.io.save(nkdir+roi_short+'.h5',data)
		
		

