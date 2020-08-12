#!/usr/bin/env python3

# Up / Down ISC graph for ALL bins
# Like SfN Poster Figure 2
# But now for all ROIs in Yeo atlas

import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import zscore
from ISC_settings import *

ROIpath = ISCpath+'shuff_Yeo/'
figdir = figurepath+'up_down/'
ROIl = [roi for roi in glob.glob(ROIpath+'*h5') if roi.split('/')[-1][:-3] not in [fig.split('/')[-1][:-4] for fig in glob.glob(figdir+'*png')]]

task = 'DM'
n_time = 750
bins = np.arange(nbinseq)
nbins = len(bins)

plt.rcParams.update({'font.size': 15})
xticks = [str(int(round(eqbins[i])))+\
		  ' - '+str(int(round(eqbins[i+1])))+' y.o.' for i in range(len(eqbins)-1)]

for roi in tqdm.tqdm(ROIl):
	roi_short = roi.split('/')[-1][:-3]
	roidict = dd.io.load(roi)
	vall = roidict[task]['vall']
	n_vox = len(vall)
	hemi = roi_short[0]
	ISC_w = np.zeros((nbins,n_vox))
	for b in range(nbins):
		if b in np.arange(1,4):
			subl = [[],[]]
			for i in [0,1]:
				subg = [ageeq[i][1][b][idx] for idx in np.random.choice(lenageeq[i][b],divmod(minageeq[i],2)[0]*2,replace=False)]
				subl[0].extend(subg[:divmod(minageeq[i],2)[0]])
				subl[1].extend(subg[divmod(minageeq[i],2)[0]:])
			groups = np.zeros((2,n_vox,n_time),dtype='float16')
			for h in [0,1]: # split all or between T / F
				group = np.zeros((n_vox,n_time),dtype='float16')
				groupn = np.ones((n_vox,n_time),dtype='int')*len(subl[0])*2
				for sub in subl[h]:
					d = np.stack([v for vi,v in enumerate(dd.io.load(sub,'/'+task+'/'+hemi)) if vi in vall])
					group = np.nansum(np.stack((group,d)),axis=0)
					nanverts = np.argwhere(np.isnan(d))
					groupn[nanverts[:, 0],nanverts[:,1]] = groupn[nanverts[:,0],nanverts[:,1]]-1
				groups[h] = zscore(group/groupn,axis=1)
			ISC_w[b]=np.sum(np.multiply(groups[0],groups[1]),axis=1)/(n_time-1)
		else:
			idx = b if bin==0 else 1
			ISC_w[b] = roidict[task]['ISC_w'][0,idx]
	roidict[task]['ISC_w_all'] = ISC_w
	dd.io.save(roi,roidict)
	fdict = {'Age':[],'ISC':[],'vox':[],'corr':[],'null':[]}
	for vi in range(len(vall)):
		for b in bins:
			fdict['null'].append(np.mean(roidict[task]['ISC_w'][1:,:,vi]))
			fdict['Age'].append(xticks[b])
			fdict['ISC'].append(ISC_w[b,vi])
			fdict['vox'].append(vi)
			fdict['corr'].append(np.corrcoef([e+agespan/2 for e in eqbins[:-1]],ISC_w[:,vi])[0,1])
	color = 'tab:red' if np.mean(fdict['corr'])>0 else 'royalblue'
	nullmean = np.mean(fdict['null'])
	nullstd = np.std(fdict['null'],ddof=1).mean()
	df = pd.DataFrame(data=fdict)
	fig = plt.fill_between(np.arange(nbinseq+1)-0.5,
                    nullmean-nullstd, nullmean+nullstd,
                    alpha=0.2, edgecolor='none', facecolor='grey')
	fig = sns.swarmplot(x="Age", y="ISC", data=df,zorder=1,color=color)
	fig = sns.pointplot(x="Age", y="ISC", data=df,markers='+',join=False,color='k',ci='sd',capsize=.1, zorder=100)
	plt.xticks(rotation=30,ha="right")
	plt.tight_layout()
	plt.show()
	b1mean = np.mean(df.loc[df['Age']==xticks[0]]['ISC'])
	sig = '*' if nullmean-nullstd>b1mean or nullmean+nullstd<b1mean else ''
	fig.figure.savefig(figdir+roi_short+sig+'.png')
	fig.figure.savefig(figdir+roi_short+sig+'.eps')
	
	
	

