#!/usr/bin/env python3

# HPC univariate bump?
# 1) Sanity Check: Measure ISC in HPC
# Does HPC ISC increase with age?
# 2) Is there an HPC bump in oldest group at event boundaries?

import tqdm
import itertools
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from HMM_settings import *

# Remove subjects over max(eqbins) age:
incl_idx = [a<eqbins[-1] for a in agel]
subord = [s for si,s in enumerate(subord) if incl_idx[si]]
agel,pcl,phenol = make_phenol(subord)
agelperm = agel
phenolperm = phenol
task = 'DM'
n_time=750
nsub=40
TW = 30
TR=0.8
nsh = 1000 # number of split half iterations
grey = 211/256
colors_age = ['#edf8fb','#b3cde3','#8c96c6','#8856a7','#810f7c']
colors_ev  = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
x = np.arange(-1*TW//2,TW//2)*TR
xstim = np.arange(n_time)*TR
xticks = [str(int(round(eqbins[i])))+\
		  ' - '+str(int(round(eqbins[i+1])))+' y.o.' for i in range(len(eqbins)-1)]
xcorrx = np.concatenate([np.arange(-n_time+1,0)*TR,np.arange(n_time)*TR])
event_list = [e for e in event_list if e+TW//2<n_time]
nevent = len(event_list)

for HPC in ['HPC','aHPC','pHPC']:
	labid = 1 if 'p' in HPC else 2
	# ISC: within and between
	D = {key:{} for key in range(nbinseq)}
	bootvs = ['bootstrap','shuffle','splithalf']
	boots = len(bootvs)
	subla = {key:{key:{key:[] for key in range(nbinseq)} for key in range(nsh)} for key in bootvs}
	ISC_w_time = np.zeros((boots,nsh,nbinseq,n_time))
	ISC_w = np.zeros((boots,nsh,nbinseq))
	ISC_b_time = {key:{key:{} for key in range(nsh)} for key in bootvs}
	ISC_b = {key:{key:{} for key in range(nsh)} for key in bootvs}
	ISC_g_time = {key:{key:{} for key in range(nsh)} for key in bootvs}
	gdict = {key:{key:{'Age1':[],'Age2':[],'g_diff':[]} for key in range(nsh)} for key in bootvs}

	for bootidx,bootv in enumerate(bootvs):
		boot = True if bootv=='bootstrap' else False
		print(HPC,bootv)
		agel = agelperm
		phenol = phenolperm
		for s in tqdm.tqdm(range(nsh)): # Either permuted shuffle,  bootstrapped resample iterations, or split-half iterations
			if bootv == 'shuffle': # randomly shuffle ages:
				ageidx = np.random.permutation(len(agel))
				agel = [agel[ageidx[idx]] for idx,age in enumerate(agel)]
				phenol['sex'] = [phenol['sex'][ageidx[idx]] for idx,age in enumerate(phenol['sex'])]
			ageeq,lenageeq,minageeq = binagesubs(agel,phenol['sex'],eqbins,subord)
			groups = np.zeros((nbinseq,2,n_time),dtype='float16')
			for b in range(nbinseq):
				subl = [[],[]]
				for i in [0,1]:
					subg = [hpcprepath+ageeq[i][1][b][idx].split('/')[-1] for idx in np.random.choice(lenageeq[i][b],divmod(minageeq[i],2)[0]*2,replace=boot)]
					subl[0].extend(subg[:divmod(minageeq[i],2)[0]])
					subl[1].extend(subg[divmod(minageeq[i],2)[0]:])
				for sub in subl[0]+subl[1]:
					if bootv != 'shuffle' and sub not in D[b]:
						data = dd.io.load(sub,['/'+task+'/HPC'])[0]
						if HPC!='HPC':
							lab = dd.io.load(sub,['/'+task+'/aplab'])[0]
							data = data[lab==labid]		
						D[b][sub] = np.mean(ss.zscore(data,axis=1),axis=0)
					elif bootv == 'shuffle' and all(sub not in D[b_] for b_ in range(nbinseq)):
						realbin = np.sum([agelperm[[sub.split('/')[-1] for sub in subord].index(sub.split('/')[-1])] >= e for e in eqbins]) - 1
						data = dd.io.load(sub,['/'+task+'/HPC'])[0]
						if HPC!='HPC':
							lab = dd.io.load(sub,['/'+task+'/aplab'])[0]
							data = data[lab==labid]	
						D[realbin][sub] = np.mean(ss.zscore(data,axis=1),axis=0)
				subla[bootv][s][b].append(subl)
				for h in [0,1]: # split all or between T / F
					group = np.zeros((n_time),dtype='float16')
					groupn = np.ones((n_time),dtype='int')*nsub
					for sub in subl[h]:
						realbin = np.sum([agelperm[[sub.split('/')[-1] for sub in subord].index(sub.split('/')[-1])] >= e for e in eqbins]) - 1
						group = np.nansum(np.stack((group,D[realbin][sub])),axis=0)
						nanverts = np.argwhere(np.isnan(D[realbin][sub]))
						groupn[nanverts] = groupn[nanverts]-1
					groups[b,h] = ss.zscore(group/groupn)
				ISC_w_time[bootidx,s,b] = np.multiply(groups[b,0],groups[b,1])
				ISC_w[bootidx,s,b] = np.sum(ISC_w_time[bootidx,s,b])/(n_time-1)
			for p in itertools.combinations(range(nbinseq),2):
				p_str = str(p[0])+'_'+str(p[1])
				ISC_g_time[bootv][s][p_str] = []
				ISC_b_time[bootv][s][p_str] = []
				for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						ISC_b_time[bootv][s][p_str].append(np.multiply(groups[p[0],htmp1], groups[p[1],htmp2]))
				ISC_b[bootv][s][p_str] = [np.sum(t)/(n_time-1) for t in ISC_b_time[bootv][s][p_str]] # correlate across bins
				ISCg_time = np.sum(ISC_b_time[bootv][s][p_str],axis=0)	
				ISCg = np.sum(ISC_b[bootv][s][p_str])
				denom = np.sqrt(ISC_w[bootidx,s,p[0]])*np.sqrt(ISC_w[bootidx,s,p[1]])
				ISC_g_time[bootv][s][p_str] = ISCg_time/4/(denom)
				ISCg = ISCg/4/(denom)
				#if ISCg>1: ISCg=1
				for k in gdict[bootv][s].keys():
					ir = [0,1] if '1' in k else [1,0]
					if 'Age' in k:
						for i in ir:
							gdict[bootv][s][k].append(str(int(round(eqbins[p[i]])))+' - '+str(int(round(eqbins[p[i]+1])))+' y.o.')
						gdict[bootv][s]['g_diff'].extend([ISCg])
			if bootv == 'shuffle': # randomly shuffle ages:
				ageidx = np.random.permutation(len(agel))
				agel = [agel[ageidx[idx]] for idx,age in enumerate(agel)]
				phenol['sex'] = [phenol['sex'][ageidx[idx]] for idx,age in enumerate(phenol['sex'])]
			
	dd.io.save(ISCpath+HPC+'.h5',{'D':D,'subla':subla, 'ISC_w_time':ISC_w_time, 'ISC_w':ISC_w, 'ISC_b_time':ISC_b_time, 'ISC_b':ISC_b, 'ISC_g_time':ISC_g_time, 'gdict':gdict})
	
# look at HPC volume vs age:
sizedict = {'Subj':[],'Age':[],'HPC':[],'aHPC':[],'pHPC':[]}
for subi,sub_ in enumerate(subord):
	sizedict['Age'].append(agelperm[subi])
	sub = hpcprepath+sub_.split('/')[-1]
	sizedict['Subj'].append(sub)
	lab = dd.io.load(sub,['/'+task+'/aplab'])[0]
	sizedict['HPC'].append(len(lab))
	sizedict['aHPC'].append(np.sum(lab==1))
	sizedict['pHPC'].append(np.sum(lab==2))
dfsize = pd.DataFrame(data=sizedict)
dd.io.save(ISCpath+'HPC_vol.h5',sizedict)
for HPC in ['HPC','aHPC','pHPC']:
	r,p = ss.pearsonr(dfsize['Age'],dfsize[HPC])
	sns.set(font_scale = 2,rc={'axes.facecolor':(grey,grey,grey)})
	fig,ax=plt.subplots(figsize=(7,5))
	sns.regplot(x='Age', y=HPC, data=dfsize,color=colors_age[3]).set_title('r = '+str(np.round(r,2))+', p = '+str(np.round(p,2)))
	ax.set_xlabel('Age')
	ax.set_ylabel(HPC+' size')
	plt.rcParams['axes.xmargin'] = 0
	print(HPC,', r = '+str(np.round(r,2)),', p = '+str(np.round(p,2)))
	fig.savefig(figurepath+'HPC/'+HPC+'_size_vs_age.png', bbox_inches='tight', dpi=300)
	
# Does autocorrelation in HPC change with age?
for HPC in ['HPC','aHPC','pHPC']:
	D = dd.io.load(ISCpath+HPC+'.h5','/D')
	autocorrlagdict = {'Age':[],'correlation':[],'Time lag [s]':[],'Subj':[],'Exact Age':[]}
	for b in range(nbinseq):
		for subj,bumps in D[b].items():
				xcorrt = xcorr(bumps,bumps)
				autocorrlagdict['Subj'].extend([subj]*len(xcorrx))
				autocorrlagdict['Age'].extend([xticks[b]]*len(xcorrx))
				autocorrlagdict['Exact Age'].extend([Phenodf['Age'][Phenodf['EID'] == subj.split('/')[-1].split('.')[0].split('-')[1]].values[0]]*len(xcorrx))
				autocorrlagdict['correlation'].extend(xcorrt)
				autocorrlagdict['Time lag [s]'].extend(xcorrx)
	dfauto = pd.DataFrame(data=autocorrlagdict)
	dfauto = dfauto[abs(dfauto['Time lag [s]'])<20]
	
	sns.set(font_scale = 2,rc={'axes.facecolor':(grey,grey,grey)})
	sns.set_palette(colors_age)
	fig,ax = plt.subplots(1,1,figsize=(7,7))
	g = sns.lineplot(x='Time lag [s]', y='correlation', hue='Age', ax=ax, data=dfauto, ci=95)
	#ax.set_xlim([-10,10])
	#ax.set_xticks([-20,-10,-5,0,5,10,20])
	ax.set_xlabel('Time (seconds)')
	ax.set_ylabel('Hippocampus-cross-correlation')
	ax.legend(loc='center', bbox_to_anchor=(0.5, -0.5))
	ax.margins(x=0)
	plt.savefig(figurepath+'HPC/'+HPC+'_autocorr.png', bbox_inches='tight',dpi=300)
			





D, subla, ISC_w_time, ISC_w, ISC_b_time, ISC_b, ISC_g_time, gdict = dd.io.load(ISCpath+'HPC.h5',['/D','/subla', '/ISC_w_time', '/ISC_w', '/ISC_b_time', '/ISC_b', '/ISC_g_time', '/gdict'])

# Are within ISCs significantly greater than 0?
p_within_greater_zero = []
for b in range(nbinseq):
	p_within_greater_zero.append(np.sum(ISC_w[0,:,b]<0)/nsh)
# Is real difference between groups' ISC_w significant?
wshuffdiff = []
for p2 in range(1,nbinseq):
	wshuffdiff.extend(ISC_w[1,:,0]-ISC_w[1,:,p2])
#for p in itertools.combinations(range(nbinseq),2):
#	wshuffdiff.extend(ISC_w[1,:,p[0]]-ISC_w[1,:,p[1]])
p_within_difference = {}
for p in itertools.combinations(range(nbinseq),2):
	p_str = str(p[0])+'_'+str(p[1])
	p_within_difference[p_str] = np.sum(np.mean(ISC_w[2,:,p[0]]-ISC_w[2,:,p[1]]) < wshuffdiff)/len(wshuffdiff)
# Are any g_diff ISCs significantly below shuffle?
gshuff = []
for s in range(nsh):
	gshuff.extend(np.unique(gdict['shuffle'][s]['g_diff'][::2]))
p_g = {}
for p in itertools.combinations(range(nbinseq),2):
	Ages = []
	for a in [0,1]:
		Ages.append(str(int(round(eqbins[p[a]])))+' - '+str(int(round(eqbins[p[a]+1])))+' y.o.')
	idxs = []
	for idx in ['Age1','Age2']:
		idxs.append([ai for ai,a in enumerate(gdict['splithalf'][0][idx]) if any(aa==a for aa in Ages)])
	idx = list(set(idxs[0]).intersection(set(idxs[1])))[0]
	p_str = str(p[0])+'_'+str(p[1])
	p_g[p_str] = np.sum(np.mean(np.array([gdict['splithalf'][gi]['g_diff'][idx] for gi in range(nsh)])) > gshuff)/len(gshuff)


# e diff plot
sns.set_palette(colors_age)
edf = pd.DataFrame(columns=['Age', 's', 'ISC'])
for s in range(nsh):
	for b in range(nbinseq):
		edf = edf.append({'Age': xticks[b], 's': s, 'ISC': ISC_w[2,s,b]}, ignore_index=True)

fig, ax = plt.subplots(figsize=(5,5))
sns.swarmplot(x='Age',y='ISC',data=edf,zorder=1)
sns.pointplot(x="Age", y="ISC", data=edf,markers='+',join=False,color='k',capsize=.1, zorder=100)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize = 15)
plt.tight_layout()
plt.savefig(figurepath+'HPC/SH_ISC_w_dist.png', bbox_inches='tight')

# Make e and g diff time plots like # Univariate bump plots
# Do old ages de-sync after event boundaries relative to "baseline"?
# ISC e diff time
ISCedict = {'Event':[],'Age':[],'s':[],'ISC':[],'Time':[]}
for event in event_list:
	esec = np.round(event*TR,2)
	for b in range(nbinseq):
		for s in range(nsh):
			ISCedict['Age'].extend([xticks[b]]*len(x))
			ISCedict['Event'].extend([esec]*len(x))
			ISCedict['s'].extend([s]*len(x))
			ISCedict['ISC'].extend(ISC_w_time[2,s,b,event-TW//2:event+TW//2])
			ISCedict['Time'].extend(x)
dfe = pd.DataFrame(data=ISCedict)
sns.set(font_scale = 2)
sns.set_palette(colors_age)
fig, ax = plt.subplots(nevent,figsize=(10, 22),sharex=True)
axa = fig.add_subplot(111, frameon=False)
axa.set_ylabel('ISC', fontsize=35,labelpad=60)
axa.set_xticks([])
axa.set_yticks([])
for ei,event in enumerate(event_list):
	esec = np.round(event*TR,2)
	g = sns.lineplot(x='Time', y='ISC',
                hue='Age', data=dfe.loc[dfe['Event'] == esec], ax=ax[ei],ci='sd')
	ax[ei].axvline(0, c='k', ls='--',lw=2)
	ax[ei].set_title('Event at '+str(esec)+' s',fontsize=30,color=colors_ev[ei])
	g.set_ylabel('')
	if ei<8: ax[ei].get_legend().remove()
ax[ei].legend(loc='lower right', bbox_to_anchor=(1.5, -0.4))
ax[ei].set_xlabel('Time [s]')
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.25,
                    wspace=0.35)
plt.savefig(figurepath+'HPC/ISC_w_time.png', bbox_inches='tight')

# ISC g time
ISCgdict = {'Event':[],'Age pair':[],'s':[],'ISC':[],'Time':[]}
for event in event_list:
	esec = np.round(event*TR,2)
	for p in itertools.combinations(range(nbinseq),2):
		for s in range(nsh):
			ISCgdict['Age pair'].extend([xticks[p[0]]+' with '+xticks[p[1]]]*len(x))
			ISCgdict['Event'].extend([esec]*len(x))
			ISCgdict['s'].extend([s]*len(x))
			ISCgdict['ISC'].extend(ISC_g_time['splithalf'][s][str(p[0])+'_'+str(p[1])][event-TW//2:event+TW//2])
			ISCgdict['Time'].extend(x)
dfg = pd.DataFrame(data=ISCgdict)
dfg = dfg[dfg['Age pair'].str.contains('16 - 19')]
sns.set(font_scale = 2)
sns.set_palette(colors_age)
fig, ax = plt.subplots(nevent,figsize=(10, 22),sharex=True)
axa = fig.add_subplot(111, frameon=False)
axa.set_ylabel('ISC', fontsize=35,labelpad=60)
axa.set_xticks([])
axa.set_yticks([])
for ei,event in enumerate(event_list):
	esec = np.round(event*TR,2)
	g = sns.lineplot(x='Time', y='ISC',
                hue='Age pair', data=dfg.loc[dfg['Event'] == esec], ax=ax[ei],ci='sd')
	ax[ei].axvline(0, c='k', ls='--',lw=2)
	ax[ei].set_title('Event at '+str(esec)+' s',fontsize=30,color=colors_ev[ei])
	g.set_ylabel('')
	if ei<8: ax[ei].get_legend().remove()
ax[ei].legend(loc='lower right', bbox_to_anchor=(1.9, -0.5))
ax[ei].set_xlabel('Time [s]')
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.25,
                    wspace=0.35)
plt.savefig(figurepath+'HPC/ISC_g_time.png', bbox_inches='tight')

# Facet plot of g_diff:
sns.set()
sns.set(font_scale = 2)
g = sns.relplot(x="Time", y="ISC", col="Age pair", row='Event', kind="line", ci='sd', data=dfg)
g.set_titles('Event at {row_name} s\n{col_name}')
g.set_xlabels("Time [s]")
g.savefig(figurepath+'HPC/ISC_g_time_facetplot.png')

# Regular (and z-scored) timecourse of g_diff (e_diff), with event boundaries
ISCgtimedict = {'Age pair':[],'s':[],'ISC':[],'z-scored ISC':[],'Time':[]}
for p in itertools.combinations(range(nbinseq),2):
	if max(range(nbinseq)) in p:
		for s in range(nsh):
			ISCgtimedict['Age pair'].extend([xticks[p[0]]+' with '+xticks[p[1]]]*len(xstim))
			ISCgtimedict['s'].extend([s]*len(xstim))
			ISCgtimedict['ISC'].extend(ISC_g_time['splithalf'][s][str(p[0])+'_'+str(p[1])])
			ISCgtimedict['z-scored ISC'].extend(ss.zscore(ISC_g_time['splithalf'][s][str(p[0])+'_'+str(p[1])]))
			ISCgtimedict['Time'].extend(xstim)
dfgtime = pd.DataFrame(data=ISCgtimedict)

sns.set_palette(colors_age)
fig,axes = plt.subplots(2,1,figsize=(20,5),sharex=True)
ISCls = ['ISC','z-scored ISC']
ymax = [20,1]
for ai,ax in enumerate(axes):
	g = sns.lineplot(x='Time',y=ISCls[ai],hue='Age pair',ci='sd', ax=ax, data=dfgtime)
	for e in event_list:
		ax.axvline(e*TR,ymin=0,ymax=20, c='k', ls='--',lw=2)
	ax.set_ylim([-0.5,ymax[ai]])
	ax.set_xlim([0,max(xstim)])
	if ai==0: ax.get_legend().remove()
	if ai==1: ax.set_xlabel('Time [s]')
ax.legend(loc='lower right', bbox_to_anchor=(1.4, -0.3))
plt.savefig(figurepath+'HPC/ISC_g_time_timecourse.png', bbox_inches='tight')

#e_diff timecourse:
ISCetimedict = {'Age':[],'s':[],'ISC':[],'z-scored ISC':[],'Time':[]}
for b in range(nbinseq):
	for s in range(nsh):
		ISCetimedict['Age'].extend([xticks[b]]*len(xstim))
		ISCetimedict['s'].extend([s]*len(xstim))
		ISCetimedict['ISC'].extend(ISC_w_time[2,s,b])
		ISCetimedict['z-scored ISC'].extend(ss.zscore(ISC_w_time[2,s,b]))
		ISCetimedict['Time'].extend(xstim)
dfetime = pd.DataFrame(data=ISCetimedict)

fig,axes = plt.subplots(2,1,figsize=(20,5),sharex=True)
ISCls = ['ISC','z-scored ISC']
ymax = [20,1]
for ai,ax in enumerate(axes):
	g = sns.lineplot(x='Time',y=ISCls[ai],hue='Age',ci='sd', ax=ax, data=dfetime)
	for e in event_list:
		ax.axvline(e*TR,ymin=0,ymax=20, c='k', ls='--',lw=2)
	ax.set_ylim([-0.5,ymax[ai]])
	ax.set_xlim([0,max(xstim)])
	if ai==0: ax.get_legend().remove()
	if ai==1: ax.set_xlabel('Time [s]')
ax.legend(loc='lower right', bbox_to_anchor=(1.2, -0.3))
plt.savefig(figurepath+'HPC/ISC_w_time_timecourse.png', bbox_inches='tight')

# Lagged correlation between age-groups:
# g_diff and e_diff
glagdict = {'Age Pair':[],'correlation':[],'Time lag [s]':[],'s':[]}
for p in itertools.combinations(range(nbinseq),2):
	if 4 in p:
		for s in range(nsh):
			ISC = ISC_g_time['splithalf'][s][str(p[0])+'_'+str(p[1])]
			xcorrt = np.correlate(ISC,ISC,'full')#xcorr(ISC,ISC)
			glagdict['Age Pair'].extend([xticks[p[0]]+' with '+xticks[p[1]]]*len(xcorrx))
			glagdict['s'].extend([s]*len(xcorrx))
			glagdict['correlation'].extend(xcorrt)
			glagdict['Time lag [s]'].extend(xcorrx)
dfglag = pd.DataFrame(data=glagdict)
fig,ax = plt.subplots(1,1,figsize=(5,5))
g = sns.lineplot(x='Time lag [s]', y='correlation',
                hue='Age Pair', ax=ax, data=dfglag[abs(dfglag['Time lag [s]'])<10], ci='sd')
ax.legend(loc='center', bbox_to_anchor=(0.5, -0.3))
plt.savefig(figurepath+'HPC/ISC_xcorr_g_nonorm.png', bbox_inches='tight')

wlagdict = {'Age':[],'correlation':[],'Time lag [s]':[],'s':[]}
for b in range(nbinseq-1):
	for s in range(nsh):
		xcorrt = np.correlate(ISC_w_time[2,s,b],ISC_w_time[2,s,b],"full")#xcorr(ISC_w_time[2,s,b],ISC_w_time[2,s,b])
		wlagdict['Age'].extend([xticks[b]]*len(xcorrx))
		wlagdict['s'].extend([s]*len(xcorrx))
		wlagdict['correlation'].extend(xcorrt)
		wlagdict['Time lag [s]'].extend(xcorrx)
dfelag = pd.DataFrame(data=wlagdict)
sns.set_palette(colors_age)
fig,ax = plt.subplots(1,1,figsize=(5,5))
g = sns.lineplot(x='Time lag [s]', y='correlation',
                hue='Age', ax=ax, data=dfelag[abs(dfelag['Time lag [s]'])<10], ci='sd')
ax.legend(loc='center', bbox_to_anchor=(0.5, -0.3))
plt.savefig(figurepath+'HPC/ISC_xcorr_within_nonorm.png', bbox_inches='tight')

# Is there a lag in HPC time course of younger kids?
bumplagdict = {'Age':[],'correlation':[],'Time lag [s]':[]}
for b in range(nbinseq-1):
	bumps = np.mean([d for d in D[b].values()],axis=0)
	xcorrt = xcorr(bumps,bumps)#np.correlate(bumps,bumps,"full")#
	bumplagdict['Age'].extend([xticks[b]]*len(xcorrx))
	bumplagdict['correlation'].extend(xcorrt)
	bumplagdict['Time lag [s]'].extend(xcorrx)
dfbumplag = pd.DataFrame(data=bumplagdict)
sns.set_palette(colors_age)
fig,ax = plt.subplots(1,1,figsize=(5,5))
g = sns.lineplot(x='Time lag [s]', y='correlation',
                hue='Age', ax=ax, data=dfbumplag[abs(dfbumplag['Time lag [s]'])<10], ci='sd')
ax.legend(loc='center', bbox_to_anchor=(0.5, -0.3))
plt.savefig(figurepath+'HPC/bump_xcorr_norm.png', bbox_inches='tight')

# Does HPC bump correlate across all time?
for p in itertools.combinations(range(nbinseq),2):
	bumps = []
	for b in p:
		bumps.append(np.mean([d for d in D[b].values()],axis=0))
	print(p)
	r,pval = ss.pearsonr(bumps[0],bumps[1])
	print(r,pval)
	
	
# g_diff plot
plt.rcParams.update({'font.size': 15})
mask = np.zeros((nbinseq,nbinseq))
mask[np.triu_indices_from(mask)] = True
g_diff_mean = {'Age1':gdict['splithalf'][0]['Age1'],'Age2':gdict['splithalf'][0]['Age2'],'g_diff':np.mean(np.array([gdict['splithalf'][gi]['g_diff'] for gi in range(nsh)]),axis=0)}
df = pd.DataFrame(data=g_diff_mean).pivot("Age1", "Age2", "g_diff")
cols = df.columns.tolist()
df = df[cols[-2:]+cols[:-2]]
df = df.reindex(cols[-2:]+cols[:-2])
with sns.axes_style("white"):
	ax = sns.heatmap(df, mask=mask, square=True,cbar_kws={'label': 'g diff ISC'},cmap='viridis',vmax=1)
ax.set_xlabel(None)
ax.set_ylabel(None)
plt.xticks(rotation=30,ha="right")
plt.yticks(rotation=30)
plt.tight_layout()
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!
plt.savefig(figurepath+'HPC/g_diff.png', bbox_inches='tight')


eseclist = []
bumpdict = {'Age':[],'Event':[],'Activity':[],'Time':[],'Subj':[]}
# Univariate bump
plt.rcParams.update({'font.size': 30})
x = np.arange(-1*TW//2,TW//2)*TR
bumps   = np.zeros((nbinseq,nevent,TW))
bumpstd = np.zeros((nbinseq,nevent,TW))
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
		bumpi = [d[e-TW//2:e+TW//2] for d in D[bi].values()]
		for sub,d in D[bi].items():
			bumpdict['Subj'].extend([sub]*len(x))
			bumpdict['Age'].extend([xticks[bi]]*len(x))
			bumpdict['Event'].extend([esec]*len(x))
			bumpdict['Activity'].extend(d[e-TW//2:e+TW//2])
			bumpdict['Time'].extend(x)
		bumps[bi,ei]   = np.mean(bumpi,axis=0)
		bumpstd[bi,ei] = np.std (bumpi,axis=0)
		lab = 'Event at '+str(esec)+' s'
		ax[bi].errorbar(x, bumps[bi,ei], yerr=bumpstd[bi,ei],color=colors_ev[ei],label=lab)
	ax[bi].errorbar(x, np.mean(bumps[bi],axis=0), yerr=np.mean(bumpstd[bi],axis=0), c='k',ls='--',label='Avg')
lgd = ax[bi].legend(loc='lower right', bbox_to_anchor=(1.7, -0.5))
#plt.savefig(figurepath+'HPC/bump.png', bbox_inches='tight')
eseclist = np.unique(eseclist)

# HPC bump plots like ISC_time plots: (Grouped by Event)
dfbump = pd.DataFrame(data=bumpdict)
sns.set(font_scale = 2)
sns.set_palette(colors_age)
fig, ax = plt.subplots(nevent,figsize=(10, 25),sharex=True)
axa = fig.add_subplot(111, frameon=False)
axa.set_ylabel('Boundary-Triggered\nHippocampal Activity', fontsize=35,labelpad=95)
axa.set_xticks([])
axa.set_yticks([])
for ei,esec in enumerate(eseclist):
	g = sns.lineplot(x='Time', y='Activity',
                hue='Age', data=dfbump.loc[dfbump['Event'] == esec], ax=ax[ei])#,ci='sd')
	ax[ei].axvline(0, c='k', ls='--',lw=2)
	ax[ei].set_title('Event at '+str(esec)+' s',fontsize=30,color=colors_ev[ei])
	ax[ei].set_xlim([min(x),max(x)])
	g.set_ylabel('')
	if ei<nevent-1: ax[ei].get_legend().remove()
ax[ei].legend(loc='lower right', bbox_to_anchor=(1.4, -0.4))
ax[ei].set_xlabel('Time [s]')
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.25,
                    wspace=0.35)
plt.savefig(figurepath+'HPC/bump_Event_group.png', bbox_inches='tight')

# bump significance test:
dfbump[['One Sample t', 'One Sample p', 'Two Sample t', 'Two Sample p', 'One Sample < 0.05', 'Two Sample < 0.05', 'Pre-Event Cummulative Sum', 'Post-Event Cummulative Sum', 'Exact Age']] = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], index=dfbump.index)
for b in tqdm.tqdm(dfbump['Age'].unique()):
	for e in dfbump['Event'].unique():
		bedf = dfbump[(dfbump['Age'] == b) & (dfbump['Event'] == e)]
		subjl = bedf['Subj'].unique()
		for s in subjl:
			dfbump['Exact Age'][dfbump['Subj'] == s] = Phenodf['Age'][Phenodf['EID'] == s.split('/')[-1].split('.')[0].split('-')[1]].values[0]
			dfbump['Pre-Event Cummulative Sum'][(dfbump['Subj'] == s) & (dfbump['Event'] == e)] = bedf[(bedf['Subj'] == s) & (bedf['Time'].between(-8,-0.5))]['Activity'].sum()
			dfbump['Post-Event Cummulative Sum'][(dfbump['Subj'] == s) & (dfbump['Event'] == e)] = bedf[(bedf['Subj'] == s) & (bedf['Time'].between(0.5,8))]['Activity'].sum()
		t1,p1 = ss.ttest_1samp(dfbump['Post-Event Cummulative Sum'][(dfbump['Age'] == b) & (dfbump['Event'] == e)].unique(),0.0)
		t2,p2 = ss.ttest_rel(dfbump['Post-Event Cummulative Sum'][(dfbump['Age'] == b) & (dfbump['Event'] == e)].unique(), dfbump['Pre-Event Cummulative Sum'][(dfbump['Age'] == b) & (dfbump['Event'] == e)].unique())
		s1='*' if p1<0.05 else ''
		s2='*' if p2<0.05 else ''
		dfbump['One Sample t'][(dfbump['Age'] == b) & (dfbump['Event'] == e)] = t1
		dfbump['One Sample p'][(dfbump['Age'] == b) & (dfbump['Event'] == e)] = p1
		dfbump['Two Sample t'][(dfbump['Age'] == b) & (dfbump['Event'] == e)] = t2
		dfbump['Two Sample p'][(dfbump['Age'] == b) & (dfbump['Event'] == e)] = p1
		dfbump['One Sample < 0.05'][(dfbump['Age'] == b) & (dfbump['Event'] == e)] = s1
		dfbump['Two Sample < 0.05'][(dfbump['Age'] == b) & (dfbump['Event'] == e)] = s2
		
dd.io.save(ISCpath+'HPC.h5',{'D':D,'subla':subla, 'ISC_w_time':ISC_w_time, 'ISC_w':ISC_w, 'ISC_b_time':ISC_b_time, 'ISC_b':ISC_b, 'ISC_g_time':ISC_g_time, 'gdict':gdict, 'dfbump':dfbump})
		
# ANOVA on cum sum:
#from statsmodels.stats.anova import AnovaRM
dfbumptemp = dfbump[dfbump['Time'] == 0]
sns.set()
sns.set_palette(colors_age)
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
fig, ax = plt.subplots(figsize=(5,5))
sns.pointplot(data=dfbumptemp, x='Event', y='Post-Event Cummulative Sum', hue='Age', dodge=True, capsize=.1, errwidth=1)
ax.legend(loc='lower right', bbox_to_anchor=(1.4, -0.3))
plt.xticks(rotation=30,ha="right")
plt.savefig(figurepath+'HPC/Bump_ANOVA.png', bbox_inches='tight')

dfbumptemp['Post-Event Cummulative Sum mean'] = np.nan
dfAge = dfbumptemp.groupby('Subj')['Post-Event Cummulative Sum'].mean()
for subj in dfbumptemp['Subj'].unique():
	dfbumptemp['Post-Event Cummulative Sum mean'][dfbumptemp['Subj'] == subj] = dfAge[subj]
dfbumpmean = dfbumptemp[dfbumptemp['Event'] == dfbumptemp['Event'].unique()[0]]

r,p = ss.pearsonr(dfbumpmean['Exact Age'],dfbumpmean["Post-Event Cummulative Sum mean"])
fig,ax=plt.subplots()
sns.regplot(x='Exact Age', y="Post-Event Cummulative Sum mean", data=dfbumpmean).set_title('r = '+str(np.round(r,2))+', p = '+str(np.round(p,2)))
fig.savefig(figurepath+'HPC/Age_vs_postbump')

minmax = [min(dfbumptemp['Exact Age']),max(dfbumptemp['Exact Age'])]
sns.set(font_scale = 1)
fig, ax = plt.subplots(nevent,figsize=(5, 22),sharex=True)
axa = fig.add_subplot(111, frameon=False)
axa.set_ylabel("Post-Event Cummulative Sum", fontsize=20,labelpad=70)
axa.set_xlabel("Age", fontsize=20,labelpad=40)
axa.set_xticks([])
axa.set_yticks([])
for ei,e in enumerate(dfbumptemp["Event"].unique()):
	dftemp = dfbumptemp[dfbumptemp['Event'] == e]
	sns.regplot(x=dftemp['Exact Age'], y=dftemp["Post-Event Cummulative Sum"], color=colors_ev[ei],ax=ax[ei])
	r,p = ss.pearsonr(dftemp['Exact Age'],dftemp["Post-Event Cummulative Sum"])
	ax[ei].set_title('Event at '+str(e)+' s\nr = '+str(np.round(r,2))+', p = '+str(np.round(p,2)),fontsize=20)
	ax[ei].set_xlim(minmax)
	ax[ei].set_ylabel('')
	ax[ei].set_xlabel('')
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.55,
                    wspace=0.35)
fig.savefig(figurepath+'HPC/Age_vs_postbump_perevent')
	

import statsmodels.api as sm
import statsmodels.formula.api as smf
dfbumptemp=dfbumptemp.rename(columns={"Post-Event Cummulative Sum":'Post','Exact Age':'Age_e'})
md = smf.mixedlm("Post ~ Age", dfbumptemp, groups=dfbumptemp["Event"])
mdf = md.fit()
print(mdf.summary())
# get ANOVA table as R like output
from statsmodels.formula.api import ols
# Ordinary Least Squares (OLS) model
model = ols('Post ~ C(Age)', data=dfbumptemp).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
# stats f_oneway functions takes the groups as input and returns F and P-value
fvalue, pvalue = ss.f_oneway(dfbumptemp['Post'][dfbump['Age'] == '5 - 8 y.o.'], dfbumptemp['Post'][dfbump['Age'] == '8 - 11 y.o.'], dfbumptemp['Post'][dfbump['Age'] == '11 - 13 y.o.'], dfbumptemp['Post'][dfbump['Age'] == '13 - 16 y.o.'],dfbumptemp['Post'][dfbump['Age'] == '16 - 19 y.o.'])
print(fvalue, pvalue)



g = sns.FacetGrid(dfbumptemp,row='Event')
g = g.map(plt.scatter, 'Exact Age', 'Post-Event Cummulative Sum')
	
	
		
# dfbump facetgrid with tvals, and pvals:
sns.set()
sns.set(font_scale = 2)
g = sns.relplot(x="Time", y="Activity", col="Event", row='Age', kind="line", hue='One Sample < 0.05', data=dfbump)#, ci='sd')
for bi,b in enumerate(dfbump['Age'].unique()):
	for ei,e in enumerate(dfbump['Event'].unique()):
		bedf = dfbump[(dfbump['Age'] == b) & (dfbump['Event'] == e)]
		strroot = 'Event at ' + str(e) + 's\n' + b
		t1 = bedf['One Sample t'].iloc[0]
		p1 = bedf['One Sample p'].iloc[0]
		t2 = bedf['Two Sample t'].iloc[0]
		p2 = bedf['Two Sample p'].iloc[0]
		s1 = bedf['One Sample < 0.05'].iloc[0]
		s2 = bedf['Two Sample < 0.05'].iloc[0]
		if s1 == '*' and s2 == '*':
			g.axes[bi][ei].set_title(strroot + '\n one samp: t='+str(np.round(t1,2))+', p=' + str(np.round(p1,2))+ s1 + '\n two samp: t=' + str(np.round(t2,2)) + ', p =' + str(np.round(p2,2)) + s2)
		elif s1 == '*':
			g.axes[bi][ei].set_title(strroot + '\n one samp: t='+str(np.round(t1,2))+', p=' + str(np.round(p1,2))+ s1 + '\n two samp: t=' + str(np.round(t2,2)))
		elif s2 == '*':
			g.axes[bi][ei].set_title(strroot + '\n one samp: t='+str(np.round(t1,2))+ '\n two samp: t=' + str(np.round(t2,2)) + ', p =' + str(np.round(p2,2)) + s2)
		else:
			g.axes[bi][ei].set_title(strroot + '\n one samp: t='+str(np.round(t1,2))+ '\n two samp: t=' + str(np.round(t2,2)))
g.set_xlabels("Time [s]")	
g._legend.remove()
g.fig.tight_layout()
g.savefig(figurepath+'HPC/bump_facetplot.png')

		


bumptime = dfbump.Time.unique()[22] # ~ 5 seconds = 22
dfbump = dfbump.loc[dfbump['Time'] == bumptime]
fig, ax = plt.subplots(nevent,figsize=(10, 22),sharex=True)
axa = fig.add_subplot(111, frameon=False)
axa.set_ylabel('Boundary-Triggered\nHippocampal Activity', fontsize=35,labelpad=100)
axa.set_xticks([])
axa.set_yticks([])
for ei,event in enumerate(eseclist):#[44.8,195.2,354.4,435.2]): # events with biggest bump in oldest group:
	ax[ei].set_title('Event at '+str(event)+' s',fontsize=30,color=colors_ev[ei])
	sns.swarmplot(x="Age", y="Activity", data=dfbump.loc[dfbump['Event'] == event],zorder=1,ax=ax[ei])
	sns.pointplot(x="Age", y="Activity", data=dfbump.loc[dfbump['Event'] == event],markers='+',join=False,color='k',ci='sd',capsize=.1, zorder=100,ax=ax[ei])
	if ei!=8: ax[ei].set_xticks([], [])
	#ax[ei].set_yticks([-0.25,0,0.25])
	#ax[ei].set_yticklabels([-0.25,0,0.25], fontsize = 25)
	ax[ei].set_ylabel('')
	ax[ei].set_xlabel('')
plt.setp(ax[ei].get_xticklabels(), rotation=30, ha="right", fontsize = 25)
plt.tight_layout()
plt.savefig(figurepath+'HPC/bump_diff.png', bbox_inches='tight')





