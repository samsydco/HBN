#!/usr/bin/env python3

# HPC univariate bump?
# 1) Sanity Check: Measure ISC in HPC
# Does HPC ISC increase with age?
# 2) Is there an HPC bump in oldest group at event boundaries?

import itertools
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from ISC_settings import *
from scipy.stats import zscore
event_list = [56,206,244,343,373,404,443,506,544]
ev_bump = [56,244,443,544]
ev_idx = [ei for ei,e in enumerate(event_list) if e in ev_bump]
task = 'DM'
n_time=750
nsub=40
TW = 30
TR=0.8

# ISC: within and between
subla = []
D = np.empty((nbinseq,nsub,n_time),dtype='float16')
groups = np.zeros((nbinseq,2,n_time),dtype='float16')
ISC_w = np.zeros(nbinseq)
ISC_b = {}
gdict = {'Age1':[],'Age2':[],'g_diff':[]}
for b in range(nbinseq):
	subl = [[],[]]
	for i in [0,1]:
		subg = [hpcprepath+ageeq[i][1][b][idx].split('/')[-1] for idx in np.random.choice(lenageeq[i][b],divmod(minageeq[i],2)[0]*2,replace=False)]
		subl[0].extend(subg[:divmod(minageeq[i],2)[0]])
		subl[1].extend(subg[divmod(minageeq[i],2)[0]:])
	for subi,sub in enumerate(subl[0]+subl[1]):
		D[b,subi] = np.mean(dd.io.load(sub,['/'+task+'/HPC'])[0],axis=0)
	subla.append(subl)
	for h in [0,1]: # split all or between T / F
		subl = np.arange(nsub//2) if h==0 else np.arange(nsub//2,nsub)
		group = np.zeros((n_time),dtype='float16')
		groupn = np.ones((n_time),dtype='int')*nsub
		for subi in subl:
			group = np.nansum(np.stack((group,D[b,subi])),axis=0)
			nanverts = np.argwhere(np.isnan(d))
			groupn[nanverts] = groupn[nanverts]-1
		groups[b,h] = zscore(group/groupn)
	ISC_w[b]=np.sum(np.multiply(groups[b,0],groups[b,1]))/(n_time-1)
for p in itertools.combinations(range(nbinseq),2):
	ISC_b[str(p[0])+'_'+str(p[1])] = []
	for htmp1 in [0,1]:
		for htmp2 in [0,1]:
			ISC_b[str(p[0])+'_'+str(p[1])].append(np.sum(np.multiply(groups[p[0], htmp1],groups[p[1],htmp2]))/(n_time-1)) # correlate across bins
	ISCg = np.sum(ISC_b[str(p[0])+'_'+str(p[1])])	
	ISCg = ISCg/4/(np.sqrt(ISC_w[p[0]])*np.sqrt(ISC_w[p[1]]))
	for k in gdict.keys():
		ir = [0,1] if '1' in k else [1,0]
		if 'Age' in k:
			for i in ir:
				gdict[k].append(str(int(round(eqbins[p[i]])))+' - '+str(int(round(eqbins[p[i]+1])))+' y.o.')
			gdict['g_diff'].extend([ISCg])
			
dd.io.save(ISCpath+'HPC/',{'D':D, 'groups':groups, 'subla':subla, 'ISC_w':ISC_w, 'ISC_b':ISC_b, 'gdict':gdict})

# g_diff plot
mask = np.zeros((nbinseq,nbinseq))
mask[np.triu_indices_from(mask)] = True
df = pd.DataFrame(data=gdict).pivot("Age1", "Age2", "g_diff")
cols = df.columns.tolist()
df = df[cols[-2:]+cols[:-2]]
df = df.reindex(cols[-2:]+cols[:-2])
with sns.axes_style("white"):
	ax = sns.heatmap(df, mask=mask, square=True,cbar_kws={'label': 'g diff ISC'},cmap='viridis')#,vmin=0.7,vmax=0.9)
ax.set_xlabel(None)
ax.set_ylabel(None)
plt.xticks(rotation=30,ha="right")
plt.yticks(rotation=30)
plt.tight_layout()
plt.savefig(figurepath+'HPC/g_diff.png', bbox_inches='tight')

bumpdict = {'Age':[],'Event':[],'Activity':[],'Time':[]}
# Univariate bump
plt.rcParams.update({'font.size': 30})
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
x=np.arange(-1*TW//2,TW//2)*TR
bumps   = np.zeros((nbinseq,len(event_list),TW))
bumpstd = np.zeros((nbinseq,len(event_list),TW))
fig, ax = plt.subplots(nbinseq,figsize=(10, 10),sharex=True)
axa = fig.add_subplot(111, frameon=False)
axa.set_xlabel('Seconds from boundary', fontsize=35,labelpad=20)
axa.set_ylabel('Boundary-Triggered\nHippocampal Activity', fontsize=35,labelpad=40)
axa.set_xticks([-10,-5,0,5,10])
axa.set_yticks([])
for bi in range(nbinseq):
	Agelab = str(int(round(eqbins[bi])))+\
		  ' - '+str(int(round(eqbins[bi+1])))+' y.o.'
	ax[bi].set_title(Agelab)
	ax[bi].axvline(0, c='k', ls='--')
	ax[bi].set_yticks([-.15,0,.15])
	for ei,e in enumerate(event_list):
		#bumpi = zscore(D[:,e-TW//2:e+TW//2],axis=1)
		bumpi = D[bi,:,e-TW//2:e+TW//2]
		for subi in range(nsub):
			bumpdict['Age'].extend([Agelab]*len(x))
			bumpdict['Event'].extend([np.round(e*TR,2)]*len(x))
			bumpdict['Activity'].extend(bumpi[subi])
			bumpdict['Time'].extend(x)
		bumps[bi,ei]   = np.mean(bumpi,axis=0)
		bumpstd[bi,ei] = np.std (bumpi,axis=0)
		lab = 'Event at '+str(np.round(e*TR,2))+' s'
		ax[bi].errorbar(x, bumps[bi,ei], yerr=bumpstd[bi,ei],color=colors[ei],label=lab)
	ax[bi].errorbar(x, np.mean(bumps[bi],axis=0), yerr=np.mean(bumpstd[bi],axis=0), c='k',ls='--',label='Avg')
lgd = ax[bi].legend(loc='lower right', bbox_to_anchor=(1.7, -0.7))
plt.savefig(figurepath+'HPC/bump.png', bbox_inches='tight')


df = pd.DataFrame(data=bumpdict)
bumptime = df.Time.unique()[22] # ~ 5 seconds
df = df.loc[df['Time'] == bumptime]
fig, ax = plt.subplots(4,figsize=(10, 10),sharex=True)
for ei,event in enumerate([44.8,195.2,354.4,435.2]): # events with biggest bump in oldest group:
	ax[ei].set_title('Event at '+str(event)+' s')
	ax[ei] = sns.swarmplot(x="Age", y="Activity", data=df.loc[df['Event'] == event],zorder=1)
	ax[ei] = sns.pointplot(x="Age", y="Activity", data=df,markers='+',join=False,color='k',ci='sd',capsize=.1, zorder=100)
plt.xticks(rotation=30,ha="right")
plt.tight_layout()
plt.savefig(figurepath+'HPC/bump_diff.png', bbox_inches='tight')




