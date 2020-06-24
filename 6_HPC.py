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

# Remove subjects over max(eqbins) age:
incl_idx = [a<eqbins[-1] for a in agel]
subord = [s for si,s in enumerate(subord) if incl_idx[si]]
agel,pcl,phenol = make_phenol(subord)
agelperm = agel
phenolperm = phenol
event_list = [56,206,244,343,373,404,443,506,544]
nevent = len(event_list)
task = 'DM'
n_time=750
nsub=40
TW = 30
TR=0.8
nsh = 1000 # number of split half iterations
colors_age = ['#edf8fb','#b3cde3','#8c96c6','#8856a7','#810f7c']
colors_ev = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
x = np.arange(-1*TW//2,TW//2)*TR
xstim = np.arange(n_time)*TR

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
	print(bootv)
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
					D[b][sub] = np.mean(zscore(dd.io.load(sub,['/'+task+'/HPC'])[0],axis=1),axis=0)
				elif bootv == 'shuffle' and all(sub not in D[b_] for b_ in range(nbinseq)):
					realbin = np.sum([agelperm[[sub.split('/')[-1] for sub in subord].index(sub.split('/')[-1])] >= e for e in eqbins]) - 1
					D[realbin][sub] = np.mean(zscore(dd.io.load(sub,['/'+task+'/HPC'])[0],axis=1),axis=0)
			subla[bootv][s][b].append(subl)
			for h in [0,1]: # split all or between T / F
				group = np.zeros((n_time),dtype='float16')
				groupn = np.ones((n_time),dtype='int')*nsub
				for sub in subl[h]:
					realbin = np.sum([agelperm[[sub.split('/')[-1] for sub in subord].index(sub.split('/')[-1])] >= e for e in eqbins]) - 1
					group = np.nansum(np.stack((group,D[realbin][sub])),axis=0)
					nanverts = np.argwhere(np.isnan(D[realbin][sub]))
					groupn[nanverts] = groupn[nanverts]-1
				groups[b,h] = zscore(group/groupn)
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
			
dd.io.save(ISCpath+'HPC.h5',{'D':D,'subla':subla, 'ISC_w_time':ISC_w_time, 'ISC_w':ISC_w, 'ISC_b_time':ISC_b_time, 'ISC_b':ISC_b, 'ISC_g_time':ISC_g_time, 'gdict':gdict})

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
xticks = [str(int(round(eqbins[i])))+\
		  ' - '+str(int(round(eqbins[i+1])))+' y.o.' for i in range(len(eqbins)-1)]
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
			ISCgtimedict['z-scored ISC'].extend(zscore(ISC_g_time['splithalf'][s][str(p[0])+'_'+str(p[1])]))
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
		ISCetimedict['z-scored ISC'].extend(zscore(ISC_w_time[2,s,b]))
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
glagdict = {'Age Pairs':[],'correlation':[],'Time lag [s]':[],'s':[]}
xcorrx = np.concatenate([np.arange(-n_time+1,0)*TR,np.arange(n_time)*TR])
p2 = (3,4)
for p1 in itertools.combinations(range(nbinseq),2):
	if max(range(nbinseq)) in p1 and p1!=p2:
		for s in range(nsh):
			ISCp1 = ISC_g_time['splithalf'][s][str(p1[0])+'_'+str(p1[1])]
			ISCp2 = ISC_g_time['splithalf'][s][str(p2[0])+'_'+str(p2[1])]
			xcorr = np.correlate(ISCp1,ISCp2,"full")
			glagdict['Age Pairs'].extend([xticks[p1[0]]+' with '+xticks[p1[1]]+' and '+xticks[p2[0]]+' with '+xticks[p2[1]]]*len(xcorrx))
			glagdict['s'].extend([s]*len(xcorrx))
			glagdict['correlation'].extend(xcorr/np.max(xcorr))
			glagdict['Time lag [s]'].extend(xcorrx)
dfglag = pd.DataFrame(data=glagdict)
fig,ax = plt.subplots(1,1,figsize=(5,5))
g = sns.lineplot(x='Time lag [s]', y='correlation',
                hue='Age Pairs', ax=ax, data=dfglag[abs(dfglag['Time lag [s]'])<10], ci='sd')
ax.legend(loc='center', bbox_to_anchor=(0.5, -0.3))
plt.savefig(figurepath+'HPC/ISC_xcorr.png', bbox_inches='tight')

wlagdict = {'Age Pair':[],'correlation':[],'Time lag [s]':[],'s':[]}
b2 = nbinseq-1
for b1 in range(nbinseq-1):
	for s in range(nsh):
		xcorr = np.correlate(ISC_w_time[2,s,b1],ISC_w_time[2,s,b2],"full")
		wlagdict['Age Pair'].extend([xticks[b1]+' with '+xticks[b2]]*len(xcorrx))
		wlagdict['s'].extend([s]*len(xcorrx))
		wlagdict['correlation'].extend(xcorr/np.max(xcorr))
		wlagdict['Time lag [s]'].extend(xcorrx)
dfelag = pd.DataFrame(data=wlagdict)
sns.set_palette(colors_age)
fig,ax = plt.subplots(1,1,figsize=(5,5))
g = sns.lineplot(x='Time lag [s]', y='correlation',
                hue='Age Pair', ax=ax, data=dfelag[abs(dfelag['Time lag [s]'])<10], ci='sd')
ax.legend(loc='center', bbox_to_anchor=(0.5, -0.3))
plt.savefig(figurepath+'HPC/ISC_xcorr_within.png', bbox_inches='tight')

	
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
bumpdict = {'Age':[],'Event':[],'Activity':[],'Time':[]}
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
		for sub in bumpi:
			bumpdict['Age'].extend([xticks[bi]]*len(x))
			bumpdict['Event'].extend([esec]*len(x))
			bumpdict['Activity'].extend(sub)
			bumpdict['Time'].extend(x)
		bumps[bi,ei]   = np.mean(bumpi,axis=0)
		bumpstd[bi,ei] = np.std (bumpi,axis=0)
		lab = 'Event at '+str(esec)+' s'
		ax[bi].errorbar(x, bumps[bi,ei], yerr=bumpstd[bi,ei],color=colors_ev[ei],label=lab)
	ax[bi].errorbar(x, np.mean(bumps[bi],axis=0), yerr=np.mean(bumpstd[bi],axis=0), c='k',ls='--',label='Avg')
lgd = ax[bi].legend(loc='lower right', bbox_to_anchor=(1.7, -0.5))
plt.savefig(figurepath+'HPC/bump.png', bbox_inches='tight')
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
	if ei<8: ax[ei].get_legend().remove()
ax[ei].legend(loc='lower right', bbox_to_anchor=(1.4, -0.4))
ax[ei].set_xlabel('Time [s]')
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.25,
                    wspace=0.35)
plt.savefig(figurepath+'HPC/bump_Event_group.png', bbox_inches='tight')


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





