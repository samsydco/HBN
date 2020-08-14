#!/usr/bin/env python3

# Determine if k at max LL is similar in 
# Youngest group (bin 0) and 
# Oldest group (bin 4)
# Is the assumption true that the number of events is equal across groups?



import tqdm
import brainiak.eventseg.event
from sklearn.model_selection import KFold
from HMM_settings import *

sametimedir = HMMpath+'shuff_5bins_train04/'
nkdir = HMMpath+'nk/'
nsub= 41
y = [0]*int(np.floor(nsub/nsplit))*4+[1]*(int(np.floor(nsub/nsplit))+1)
kf = KFold(n_splits=nsplit, shuffle=True, random_state=2)
bins = [0,4]
nbins = len(bins)

for roi in tqdm.tqdm(glob.glob(sametimedir+'*h5')):
	roi_short = roi.split('/')[-1][:-3]
	roidict = {t:{b:{} for b in bins} for t in tasks}
	for ti,task in enumerate(tasks):
		nTR_ = nTR[ti]
		taskv = dd.io.load(roi,'/'+task)
		roidict[task]['vall'] = taskv['vall']
		for b in bins:
			roidict[task][b]['D'] = taskv['bin_'+str(b)]['D']
			tune_ll = np.zeros((nsplit,len(k_list)))
			for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
				Dtrain = np.mean(roidict[task][b]['D'][Ls[0]],axis=0).T
				Dtest  = np.mean(roidict[task][b]['D'][Ls[1]],axis=0).T
				for ki,k in enumerate(k_list):
					hmm = brainiak.eventseg.event.EventSegment(n_events=k)
					hmm.fit(Dtrain)
					_, tune_ll[split,ki] = hmm.find_events(Dtest)
			roidict[task][b]['tune_ll'] = tune_ll
			roidict[task][b]['best_k'] = k_list[np.argmax(np.mean(tune_ll,0))]
	dd.io.save(nkdir+roi_short+'.h5',roidict)
		
					
	