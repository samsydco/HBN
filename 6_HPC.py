#!/usr/bin/env python3

# HPC univariate bump?
# 1) Sanity Check: Measure ISC in HPC
# Does HPC ISC increase with age?
# 2) Is there an HPC bump in oldest group at event boundaries?

import tqdm
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
nsh = 100 # number of split half iterations

# ISC: within and between
subla = {key:{key:[] for key in range(nbinseq)} for key in range(nsh)}
D = {key:{} for key in range(nbinseq)}
ISC_w_time = np.zeros((nsh,nbinseq,n_time))
ISC_w = np.zeros((nsh,nbinseq))
ISC_b_time = {key:{} for key in range(nsh)}
ISC_b = {key:{} for key in range(nsh)}
ISC_g_time = {key:{} for key in range(nsh)}
gdict = {key:{'Age1':[],'Age2':[],'g_diff':[]} for key in range(nsh)}
for s in tqdm.tqdm(range(nsh)):
	#ageeq,lenageeq,minageeq = binagesubs(agel,phenol['sex'],eqbins,subord)
	groups = np.zeros((nbinseq,2,n_time),dtype='float16')
	for b in range(nbinseq):
		subl = [[],[]]
		for i in [0,1]:
			subg = [hpcprepath+ageeq[i][1][b][idx].split('/')[-1] for idx in np.random.choice(lenageeq[i][b],divmod(minageeq[i],2)[0]*2,replace=False)]
			subl[0].extend(subg[:divmod(minageeq[i],2)[0]])
			subl[1].extend(subg[divmod(minageeq[i],2)[0]:])
		for sub in subl[0]+subl[1]:
			D[b][sub] = np.mean(zscore(dd.io.load(sub,['/'+task+'/HPC'])[0],axis=1),axis=0)
		subla[s][b].append(subl)
		for h in [0,1]: # split all or between T / F
			group = np.zeros((n_time),dtype='float16')
			groupn = np.ones((n_time),dtype='int')*nsub
			for sub in subl[h]:
				group = np.nansum(np.stack((group,D[b][sub])),axis=0)
				nanverts = np.argwhere(np.isnan(D[b][sub]))
				groupn[nanverts] = groupn[nanverts]-1
			groups[b,h] = zscore(group/groupn)
		ISC_w_time[s,b] = np.multiply(groups[b,0],groups[b,1])
		ISC_w[s,b] = np.sum(ISC_w_time[s,b])/(n_time-1)
	for p in itertools.combinations(range(nbinseq),2):
		p_str = str(p[0])+'_'+str(p[1])
		ISC_g_time[s][p_str] = []
		ISC_b_time[s][p_str] = []
		ISC_b     [s][p_str] = []
		for htmp1 in [0,1]:
			for htmp2 in [0,1]:
				ISC_b_time[s][p_str].append(np.multiply(groups[p[0],htmp1], groups[p[1],htmp2]))
				ISC_b[s][p_str].append(np.sum(ISC_b_time[s][p_str])/(n_time-1)) # correlate across bins
		ISCg_time = np.sum(ISC_b_time[s][p_str])	
		ISCg = np.sum(ISC_b[s][p_str])	
		ISC_g_time[s][p_str] = ISCg_time/4/(np.sqrt(ISC_w_time[s,p[0]])*np.sqrt(ISC_w_time[s,p[1]])
		ISCg = ISCg/4/(np.sqrt(ISC_w[s,p[0]])*np.sqrt(ISC_w[s,p[1]]))
		#if ISCg>1: ISCg=1
		for k in gdict[s].keys():
			ir = [0,1] if '1' in k else [1,0]
			if 'Age' in k:
				for i in ir:
					gdict[s][k].append(str(int(round(eqbins[p[i]])))+' - '+str(int(round(eqbins[p[i]+1])))+' y.o.')
				gdict[s]['g_diff'].extend([ISCg])
	# randomly shuffle ages:
	#ageidx = np.random.permutation(len(agel))
	#agel = [agel[ageidx[idx]] for idx,age in enumerate(agel)]
	#phenol['sex'] = [phenol['sex'][ageidx[idx]] for idx,age in enumerate(phenol['sex'])]
			
dd.io.save(ISCpath+'HPC.h5',{'D':D,'subla':subla, 'ISC_w_time':ISC_w_time, 'ISC_w':ISC_w, 'ISC_b_time':ISC_b_time, 'ISC_b':ISC_b, 'ISC_g_time':ISC_g_time, 'gdict':gdict})


# e diff plot
colors = ['#edf8fb','#b3cde3','#8c96c6','#8856a7','#810f7c']
xticks = [str(int(round(eqbins[i])))+\
		  ' - '+str(int(round(eqbins[i+1])))+' y.o.' for i in range(len(eqbins)-1)]
df = pd.DataFrame(columns=['Age', 's', 'ISC'])
for s in range(nsh):
	for b in range(nbinseq):
		df = df.append({'Age': xticks[b], 's': s, 'ISC': ISC_w[s,b]}, ignore_index=True)

fig, ax = plt.subplots(figsize=(5,5))
sns.swarmplot(x='Age',y='ISC',data=df,zorder=1,color=colors)
sns.pointplot(x="Age", y="ISC", data=df,markers='+',join=False,color='k',capsize=.1, zorder=100)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize = 15)
plt.tight_layout()
#plt.savefig(figurepath+'HPC/SH_ISC_w_dist.png', bbox_inches='tight')


colors = ['#edf8fb','#b3cde3','#8c96c6','#8856a7','#810f7c']
null = list(itertools.chain.from_iterable(ISC_w[1:,:]))
iscticks = np.round(np.arange(min(null), max(null), 0.1),2)
fig, ax = plt.subplots()
n, bins, patches = ax.hist(null, density=True, bins=30,color='darkgrey')
for bi in range(nbinseq):
	ax.axvline(ISC_w[0,bi], c=colors[bi],label=xticks[bi])
ax.set_xticks(iscticks)
ax.set_xticklabels(iscticks, fontsize = 12)
ax.set_yticklabels([0,5], fontsize = 12)
ax.set_xlabel('within group ISC',fontsize=20)
ax.set_ylabel('Count',fontsize=20)
lgd = ax.legend(loc='lower right', bbox_to_anchor=(1.5, -0.2),fontsize=15)
plt.savefig(figurepath+'HPC/ISC_w.png', bbox_inches='tight')



# g_diff plot
plt.rcParams.update({'font.size': 15})
mask = np.zeros((nbinseq,nbinseq))
mask[np.triu_indices_from(mask)] = True
g_diff_mean = {'Age1':gdict[0]['Age1'],'Age2':gdict[0]['Age2'],'g_diff':np.mean(np.array([gdict[gi]['g_diff'] for gi in range(nsh)]),axis=0)}
df = pd.DataFrame(data=g_diff_mean).pivot("Age1", "Age2", "g_diff")
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

eseclist = []
bumpdict = {'Age':[],'Event':[],'Activity':[],'Time':[]}
# Univariate bump
plt.rcParams.update({'font.size': 30})
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
x=np.arange(-1*TW//2,TW//2)*TR
bumps   = np.zeros((nbinseq,len(event_list),TW))
bumpstd = np.zeros((nbinseq,len(event_list),TW))
fig, ax = plt.subplots(nbinseq,figsize=(11, 20),sharex=True)
axa = fig.add_subplot(111, frameon=False)
axa.set_xlabel('Seconds from boundary', fontsize=35,labelpad=50)
axa.set_ylabel('Boundary-Triggered\nHippocampal Activity', fontsize=35,labelpad=110)
axa.set_xticks([])
axa.set_yticks([])
for bi in range(nbinseq):
	ax[bi].set_title(xticks[bi])
	ax[bi].axvline(0, c='k', ls='--')
	#ax[bi].set_yticks([-.15,0,.15])
	for ei,e in enumerate(event_list):
		esec = np.round(e*TR,2)
		eseclist.append(esec)
		#bumpi = zscore(D[0,:,e-TW//2:e+TW//2],axis=1)
		bumpi = np.mean(D[:,bi,:,e-TW//2:e+TW//2],axis=0)
		for subi in range(nsub):
			bumpdict['Age'].extend([xticks[bi]]*len(x))
			bumpdict['Event'].extend([esec]*len(x))
			bumpdict['Activity'].extend(bumpi[subi])
			bumpdict['Time'].extend(x)
		bumps[bi,ei]   = np.mean(bumpi,axis=0)
		bumpstd[bi,ei] = np.std (bumpi,axis=0)
		lab = 'Event at '+str(esec)+' s'
		ax[bi].errorbar(x, bumps[bi,ei], yerr=bumpstd[bi,ei],color=colors[ei],label=lab)
	ax[bi].errorbar(x, np.mean(bumps[bi],axis=0), yerr=np.mean(bumpstd[bi],axis=0), c='k',ls='--',label='Avg')
lgd = ax[bi].legend(loc='lower right', bbox_to_anchor=(1.7, -0.5))
plt.savefig(figurepath+'HPC/bump.png', bbox_inches='tight')
eseclist = np.unique(eseclist)


df = pd.DataFrame(data=bumpdict)
bumptime = df.Time.unique()[22] # ~ 5 seconds = 22
df = df.loc[df['Time'] == bumptime]
fig, ax = plt.subplots(9,figsize=(10, 22),sharex=True)
axa = fig.add_subplot(111, frameon=False)
axa.set_ylabel('Boundary-Triggered\nHippocampal Activity', fontsize=35,labelpad=100)
axa.set_xticks([])
axa.set_yticks([])
for ei,event in enumerate(eseclist):#[44.8,195.2,354.4,435.2]): # events with biggest bump in oldest group:
	ax[ei].set_title('Event at '+str(event)+' s',fontsize=30)
	sns.swarmplot(x="Age", y="Activity", data=df.loc[df['Event'] == event],zorder=1,ax=ax[ei])
	sns.pointplot(x="Age", y="Activity", data=df.loc[df['Event'] == event],markers='+',join=False,color='k',ci='sd',capsize=.1, zorder=100,ax=ax[ei])
	if ei!=8: ax[ei].set_xticks([], [])
	#ax[ei].set_yticks([-0.25,0,0.25])
	#ax[ei].set_yticklabels([-0.25,0,0.25], fontsize = 25)
	ax[ei].set_ylabel('')
	ax[ei].set_xlabel('')
plt.setp(ax[ei].get_xticklabels(), rotation=30, ha="right", fontsize = 25)
plt.tight_layout()
plt.savefig(figurepath+'HPC/bump_diff.png', bbox_inches='tight')




