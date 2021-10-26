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
task = 'DM'

for seed in tqdm.tqdm(seeds):
	if not os.path.exists(nkdir+seed): os.makedirs(nkdir+seed)
	for roi in glob.glob(roidir+seed+'/*h5'):
		roi_short = roi.split('/')[-1][:-3]
		roidict = {str(b):{} for b in bins}
		Ddict = dd.io.load(roi)
		for b in bins:
			roidict[str(b)]['D'] = Ddict[task]['bin_'+str(b)]['D']
			roidict[str(b)]['tune_ll'] = np.zeros((nsplit,len(k_list)))
			for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
				Dtrain = np.mean(roidict[str(b)]['D'][Ls[0]],axis=0).T
				Dtest  = np.mean(roidict[str(b)]['D'][Ls[1]],axis=0).T
				for ki,k in enumerate(k_list):
					hmm = brainiak.eventseg.event.EventSegment(n_events=k)
					hmm.fit(Dtrain)
					_, roidict[str(b)]['tune_ll'][split,ki] = hmm.find_events(Dtest)			
			roidict[str(b)]['best_k'] = np.mean([k_list[np.argmax(roidict[str(b)]['tune_ll'][ki])] for ki in range(kf.n_splits)])
		roidict['k_diff'] = roidict[str(4)]['best_k'] - roidict[str(0)]['best_k']
		dd.io.save(nkdir+seed+'/'+roi_short+'.h5',roidict)
	
