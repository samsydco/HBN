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
nshuff=1000

for roi in tqdm.tqdm(glob.glob(sametimedir+'*h5')):
	roi_short = roi.split('/')[-1][:-3]
	roidict = {t:{str(b):{} for b in bins} for t in tasks}
	for ti,task in enumerate(tasks):
		nTR_ = nTR[ti]
		taskv = dd.io.load(roi,'/'+task)
		roidict[task]['vall'] = taskv['vall']
		for b in bins:
			roidict[task][b]['D'] = taskv['bin_'+str(b)]['D']
		Dall = np.concatenate((roidict[task][0]['D'],roidict[task][4]['D']),axis=0)
		for b in bins:
			best_k = np.zeros(nshuff)
			tune_ll = np.zeros((nshuff+1,nsplit,len(k_list)))
			subl = np.concatenate([np.zeros(nsub),4*np.ones(nsub)])
			for shuff in nshuff+1:
				D = Dall[np.where(subl==b)[0]]
				for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
					Dtrain = np.mean(D[Ls[0]],axis=0).T
					Dtest  = np.mean(D[Ls[1]],axis=0).T
					for ki,k in enumerate(k_list):
						hmm = brainiak.eventseg.event.EventSegment(n_events=k)
						hmm.fit(Dtrain)
						_, tune_ll[shuff,split,ki] = hmm.find_events(Dtest)
				# RANDOMIZE
				subl = np.random.permutation(subl)
				best_k[shuff] = k_list[np.argmax(np.mean(tune_ll[shuff],0))]
			roidict[task][b]['tune_ll'] = tune_ll
			roidict[task][b]['best_k'] = best_k
	dd.io.save(nkdir+roi_short+'.h5',roidict)
	
task='DM'
roidict = {k:{k:[] for k in ['k','ll','roi']} for k in ['0','4']}
for roi in tqdm.tqdm(glob.glob(nkdir+'*.h5')):
	roi_short = roi.split('/')[-1][:-3]
	data = dd.io.load(roi,'/'+task)
	for b in bins:
		roidict[str(b)]['roi'].append(roi_short)
		roidict[str(b)]['k'].append(data[b]['best_k'])
		kidx = np.where(k_list==data[b]['best_k'])[0][0]
		roidict[str(b)]['ll'].append(np.mean(data[b]['tune_ll'][0,:,kidx]))

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
grey=211/256
plt.rcParams.update({'font.size': 20})
xticks = [str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))+' y.o.' for b in bins]
for lab in ['_lab','']:
	for val in ['k','ll']:
		r,p=pearsonr(roidict['0'][val], roidict['4'][val])
		fig, ax = plt.subplots(figsize=(15,15))
		ax.set_title('r:'+str(np.round(r,2))+', p:'+str(np.round(p,2)))
		minval = np.min(roidict['0'][val])
		maxval = np.max(roidict['0'][val])
		ax.plot([minval,maxval],\
				 [minval,maxval], 'k--', linewidth=2)
		ax.set_xlabel(xticks[0]+' '+val+'\'s')
		ax.set_ylabel(xticks[1]+' '+val+'\'s')
		if lab == '_lab':
			for i, txt in enumerate(roidict['0']['roi']):
				if roidict['0'][val][i]>roidict['4'][val][i]:
					ax.annotate('\n'.join(txt.split('_')[:-1]), 
								(roidict['0'][val][i], roidict['4'][val][i]),
							horizontalalignment="left",
							verticalalignment="top",color=(grey,grey,grey))
				if roidict['0'][val][i]<roidict['4'][val][i]:
					ax.annotate('\n'.join(txt.split('_')[:-1]), 
								(roidict['0'][val][i], roidict['4'][val][i]),
							horizontalalignment="right",
							verticalalignment="bottom",color=(grey,grey,grey))
		ax.scatter(roidict['0'][val], roidict['4'][val])
		fig.savefig(figurepath+'n_k/'+val+lab+'.png')
		
		
			
			
			or roidict['4'][val][i]*1.25>roidict['0'][val][i]:
			
		
	fig.savefig(figurepath+'n_k/'+val+'.png')
		
		
	
		
					
	
