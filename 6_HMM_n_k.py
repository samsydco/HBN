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
nkdir = HMMpath+'nk_moreshuff/'#'nk/'
nsub= 41
y = [0]*int(np.floor(nsub/nsplit))*4+[1]*(int(np.floor(nsub/nsplit))+1)
kf = KFold(n_splits=nsplit, shuffle=True, random_state=2)
bins = [0,4]
nbins = len(bins)
nshuff=100
task = 'DM'

for roi in tqdm.tqdm(glob.glob(sametimedir+'*h5')):
	roi_short = roi.split('/')[-1][:-3]
	roidict = {str(b):{} for b in bins}
	taskv = dd.io.load(roi,'/'+task)
	roidict['vall'] = taskv['vall']
	for b in bins:
		roidict[str(b)]['D'] = taskv['bin_'+str(b)]['D']
	Dall = [roidict['0']['D'],roidict['4']['D']]
	for bi,b in enumerate(bins):
		notbi = 1 if bi==0 else 0
		best_k = np.zeros(nshuff+1)
		tune_ll = np.zeros((nshuff+1,nsplit,len(k_list)))
		for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
			Dtrain = np.mean(Dall[bi][Ls[0]],axis=0).T
			Dtest_all  = np.concatenate((Dall[bi][Ls[1]],Dall[notbi][Ls[1]]),axis=0)
			subl = np.concatenate((np.ones(len(Ls[1])),np.zeros(len(Ls[1]))),axis=0)
			for ki,k in enumerate(k_list):
				hmm = brainiak.eventseg.event.EventSegment(n_events=k)
				hmm.fit(Dtrain)
				for shuff in range(nshuff+1):
					_, tune_ll[shuff,split,ki] = hmm.find_events(np.mean(Dtest_all[subl==1], axis=0).T)
					# RANDOMIZE
					subl = np.random.permutation(subl)
		for shuff in range(nshuff+1):
			 best_k[shuff] = np.mean([k_list[np.argmax(tune_ll[shuff,ki])] for ki in range(kf.n_splits)])
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
		
dd.io.save(HMMpath+'nk2.h5',roidict)
roidict=dd.io.load(HMMpath+'nk2.h5')
import pandas as pd
df=pd.DataFrame(roidict).T

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
color = '#8856a7'
grey=211/256
plt.rcParams.update({'font.size': 30})
xticks = [str(int(round(eqbins[b])))+' - '+str(int(round(eqbins[b+1])))+' y.o.' for b in bins]

x = df['0']; y = df['4']
r,p=pearsonr(x,y)
fig, ax = plt.subplots(figsize=(15,15))
#ax.set_title('r:'+str(np.round(r,2))+', p:'+str(np.round(p,2)))
minval = np.min(x)
maxval = np.max(x)
ax.plot([minval,maxval],\
		 [minval,maxval], 'k--', linewidth=2)
ax.set_xlabel('Number of events for\n'+xticks[0]+'\'s', fontsize=35)
ax.set_ylabel('Number of events for\n'+xticks[1]+'\'s', fontsize=35)
ax.scatter(x,y,color=color,s=100,alpha=0.25)
if lab == '': ax.set_facecolor((grey,grey,grey))
if lab == '_lab':
	for i, txt in enumerate(list(df.index)):
		if x[i]>y[i]:
			ax.annotate('\n'.join(txt.split('_')[:-1]), (x[i], y[i]),
					horizontalalignment="left",
					verticalalignment="top",color='k',fontsize=10)
		if x[i]<y[i]:
			ax.annotate('\n'.join(txt.split('_')[:-1]), (x[i], y[i]),
					horizontalalignment="right",
					verticalalignment="bottom",color='k',fontsize=10)
		
#fig.savefig(figurepath+'n_k/'+val+lab+'.png')
		

		
	
		
					
	
