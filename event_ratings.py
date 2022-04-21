#!/usr/bin/env python3

import numpy as np
import pandas as pd
import deepdish as dd
import matplotlib.pyplot as plt
from settings import *
from event_comp import *

ev_conv = child_ev_conv

ev_figpath = figurepath+'event_annotations/'
	
# Tally of annotations at each time point
time = np.arange(0,nTR)
fig, (raw_ev_annot) = plt.subplots(figsize=(60, 20))
raw_ev_annot.hist(ev_annot, bins=nTR, linewidth=20,color='k')
yticks = list(np.arange(0, max(counts)+1, 2))
raw_ev_annot.spines['right'].set_color('none')
raw_ev_annot.spines['top'].set_color('none')
raw_ev_annot.set_xticks(np.append(time[0::nTR//5],time[-1]))
raw_ev_annot.set_xticklabels([str(int(s*TR//60))+':'+str(int(s*TR%60))+'0' for s in time][0::nTR//5]+['10:00'], fontsize=80)
raw_ev_annot.set_yticks(yticks)
plt.rcParams['xtick.labelsize']=50
plt.rcParams['ytick.labelsize']=80
raw_ev_annot.set_ylabel('# annotations', fontsize=80)
raw_ev_annot.set_xlabel('Time (minutes)', fontsize=80)
plt.tight_layout()
plt.savefig(ev_figpath+'ev_annots_hist.png', bbox_inches='tight')

# counts convolved with HRF
fig, (hrf_ann) = plt.subplots(figsize=(60, 20))
hrf_ann.plot(np.arange(nTR), ev_conv, linewidth=5,color='k')
hrf_ann.spines['right'].set_color('none')
hrf_ann.spines['top'].set_color('none')
plt.rcParams['xtick.labelsize']=80
plt.rcParams['ytick.labelsize']=80
hrf_ann.set_xlabel('TR (1 TR = 0.8 seconds)', fontsize=40)
plt.savefig(ev_figpath+'hrf_conv.png', bbox_inches='tight')

# illustration of counts with HRF-covlved counts
fig, (raw_ev_annot) = plt.subplots(figsize=(60, 20))
raw_ev_annot.hist(ev_annot, bins=nTR, linewidth=20,color='k')
raw_ev_annot.plot(np.arange(nTR), ev_conv, color='k',alpha=0.5,linewidth=5)
raw_ev_annot.spines['right'].set_color('none')
raw_ev_annot.spines['top'].set_color('none')
raw_ev_annot.set_xticks(np.append(time[0::nTR//5],time[-1]))
raw_ev_annot.set_xticklabels([str(int(s*TR//60))+':'+str(int(s*TR%60))+'0' for s in time][0::nTR//5]+['10:00'], fontsize=80)
raw_ev_annot.set_yticks(yticks)
plt.rcParams['xtick.labelsize']=50
plt.rcParams['ytick.labelsize']=80
raw_ev_annot.set_ylabel('# annotations', fontsize=80)
raw_ev_annot.set_xlabel('Time (minutes)', fontsize=80)
plt.tight_layout()
plt.savefig(ev_figpath+'ev_annots_both.png', bbox_inches='tight')

from HMM_settings import *
nTR = 750
D_ = dd.io.load(ISCpath+'HPC.h5',['/D'])[0]
D = np.zeros(nTR)
nsubj = 0
for b in range(nbinseq):
	for bumps in D_[b].values():
		D=D+bumps
		nsubj+=1
D=D/nsubj

# average hippocampal activity overlaid with counts and ev_conv:
fig, (raw_ev_annot) = plt.subplots(figsize=(60, 20))
raw_ev_annot.hist(ev_annot, bins=nTR, linewidth=20,color='k')
raw_ev_annot.plot(np.arange(nTR), ev_conv, color='k',alpha=0.5,linewidth=5)
ax2 = raw_ev_annot.twinx()
ax2.plot(np.arange(nTR)[5:], D[5:], color='k',linewidth=10)
raw_ev_annot.spines['right'].set_color('none')
raw_ev_annot.spines['top'].set_color('none')
raw_ev_annot.set_xticks(np.append(time[0::nTR//5],time[-1]))
raw_ev_annot.set_xticklabels([str(int(s*TR//60))+':'+str(int(s*TR%60))+'0' for s in time][0::nTR//5]+['10:00'], fontsize=80)
raw_ev_annot.set_yticks(yticks)
plt.rcParams['xtick.labelsize']=50
plt.rcParams['ytick.labelsize']=80
raw_ev_annot.set_ylabel('# annotations', fontsize=80)
raw_ev_annot.set_xlabel('Time (minutes)', fontsize=80)
plt.tight_layout()
plt.savefig(ev_figpath+'ev_annots_HPC.png', bbox_inches='tight')
plt.savefig(ev_figpath+'ev_annots_HPC.eps', bbox_inches='tight')

# average hippocampal activity	
fig, ax = plt.subplots(figsize=(60, 20))
ax.plot(np.arange(nTR)[5:], D[5:], color='k',linewidth=10)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xticks(np.append(time[0::nTR//5],time[-1]))
ax.set_xticklabels([str(int(s*TR//60)) +':'+ str(int(s*TR%60))+'0' for s in time][0::nTR//5]+['10:00'], fontsize=80)
plt.rcParams['xtick.labelsize']=50
ax.set_xlabel('Time (minutes)', fontsize=80)
plt.tight_layout()
plt.savefig(figurepath+'HPC/HPC_timecourse.png', bbox_inches='tight')


# Compare annotations "ev_conv" with HPC bumps:
# Precidence for thinking this way comes from Aya Ben-Yakov's event saliency measure
sizedict = dd.io.load(ISCpath+'HPC_vol.h5')
sizedf = pd.DataFrame(data=sizedict)
import statsmodels.api as sm
import statsmodels.formula.api as sm2
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import KFold
colors_age = ['#FCC3A1','#F08B63','#D02941','#70215D','#311638']
xticks = [str(int(round(eqbins[i])))+\
	  ' - '+str(int(round(eqbins[i+1])))+' y.o.' for i in range(len(eqbins)-1)]
maxlag = 10
xcorrx = np.concatenate([np.arange(-maxlag,0)*TR,np.arange(maxlag+1)*TR])
nsub = nsubj
Dall = {}
bumplagdict = {'Age':[],'Time lag [s]':[],'Subj':[],'Exact_Age':[]}
for HPC in ['HPC','aHPC','pHPC']:
	D = dd.io.load(ISCpath+HPC+'.h5','/D')
	Dall[HPC] = D
	bumplagdict['correlation_'+HPC] = []
	

for b in range(nbinseq):
	for subj in Dall['HPC'][b].keys():
		bumplagdict['Subj'].extend([subj]*len(xcorrx))
		bumplagdict['Age'].extend([xticks[b]]*len(xcorrx))
		bumplagdict['Exact_Age'].extend([Phenodf['Age'][Phenodf['EID'] == subj.split('/')[-1].split('.')[0].split('-')[1]].values[0]]*len(xcorrx))
		for HPC in ['HPC','aHPC','pHPC']:
			bumplagdict['correlation_'+HPC].extend(lag_pearsonr(Dall[HPC][b][subj],ev_conv,maxlag))
		bumplagdict['Time lag [s]'].extend(xcorrx)
dfbumplag = pd.DataFrame(data=bumplagdict)

# Which time points post-0 are significantly different from zero?
times = dfbumplag['Time lag [s]'].unique()
tvals = np.zeros(len(times))
pvals = np.zeros(len(times))
for ti,tp in enumerate(times):
	tvals[ti],pvals[ti] = stats.ttest_1samp(dfbumplag[dfbumplag['Time lag [s]']==tp]['correlation_HPC'],0)
pvals = pvals*len(pvals) # Bonferroni correction
best_t_i = times[np.argmin(pvals)]
best_t = 0 # Set from whole HPC
tempdf = dfbumplag[dfbumplag['Time lag [s]']==best_t]
t2 = np.zeros((3,len(xticks)))
p2 = np.zeros((3,len(xticks)))
for hi,HPC in enumerate(['HPC','aHPC','pHPC']):
	for ai,age in enumerate(xticks):
		t2[hi,ai],p2[hi,ai] = stats.ttest_1samp(tempdf[tempdf['Age']==age]['correlation_'+HPC],0)
tempsize = tempdf.merge(sizedf, on='Subj')
# test for linear vs polynomial fit of Age vs event_correlation:
# both line and U fit equally well!
nsplit = 20
y = [0]*int(np.floor(nsub/nsplit))*(nsplit-1)+[1]*int(np.floor(nsub/nsplit)+14)
kf = KFold(n_splits=nsplit, shuffle=True, random_state=2)
for HPC in ['HPC','aHPC','pHPC']:
	sse = np.zeros((2,nsplit))
	for degi,deg in enumerate([1,2]):
		for split,Ls in enumerate(kf.split(np.arange(nsub),y)):
			coeffs = np.polyfit(tempdf['Exact_Age'].iloc[Ls[0]],
								tempdf['correlation_'+HPC].iloc[Ls[0]],deg)
			pred = np.polyval(coeffs, tempdf['Exact_Age'].iloc[Ls[1]])
			sse[degi,split] = np.sum(np.square(tempdf['correlation_'+HPC].iloc[Ls[1]]-pred))
		print('Sum of Squared Error for Age-to-'+HPC+' to event boundary correlation is: '+
			 str(np.round(np.mean(sse[degi]),3))+'+/-'+str(np.round(np.std(sse[degi]),3))+
			 ' for degree '+str(deg)+'.')
	t,p = stats.ttest_rel(sse[0],sse[1])
	print(t,p)
# Test for interaction between aHPC and pHPC correlation with ev_conv:
result = sm2.ols(formula='Exact_Age ~ correlation_aHPC + correlation_pHPC + correlation_aHPC * correlation_pHPC', data=tempdf).fit()
print(result.summary()) #no interaction!


for HPC in ['HPC','aHPC','pHPC']:
	OLS_model = sm.OLS(tempsize['correlation_'+HPC],tempsize[HPC]).fit()  # training the model
	residual_values = OLS_model.resid # residual values
	r3,p3 = stats.pearsonr(tempsize['Exact_Age'],tempsize['correlation_'+HPC])

	X = sm.add_constant(tempsize[['Exact_Age',HPC]]) # adding a constant
	model = sm.OLS(tempsize['correlation_'+HPC], X).fit()
	predictions = model.predict(X)
	print_model = model.summary()
	print(print_model)

for HPC in ['HPC','aHPC','pHPC']:
	r,p = stats.pearsonr(tempdf['Exact_Age'],tempdf['correlation_'+HPC])
	sns.set(font_scale = 2,style="ticks")
	fig,ax=plt.subplots(figsize=(7,5))
	sns.scatterplot(x='Exact_Age',y="correlation_"+HPC,hue='Age', data=tempdf,palette=colors_age,ax=ax,linewidth=0,alpha = 0.7,legend=False,s=50)
	sns.regplot(x='Exact_Age', y="correlation_"+HPC, data=tempdf, scatter=False, ax=ax,color=colors_age[2])
	ax.set_xlabel('Age')
	ax.set_ylabel('Hippocampus-to-event\ncorrelation')
	plt.rcParams['axes.xmargin'] = 0
	print(HPC,', r = '+str(np.round(r,2)),', p = '+str(np.round(p,5)))
	fig.savefig(figurepath+'HPC/'+HPC+'_Age_vs_bump_xcorr_ev_conv.png', bbox_inches='tight', dpi=300)

for HPC in ['HPC','aHPC','pHPC']:
	# no correlation between size and event-response!!
	r2,p2 = stats.pearsonr(tempsize[HPC],tempsize['correlation_'+HPC])
	# plot indicating size
	fig,ax=plt.subplots(figsize=(7,5))
	sns.scatterplot(x='Exact_Age', y="correlation_"+HPC, hue=HPC, size=HPC,linewidth=0,alpha=.5, 
			data=tempsize, ax=ax)
	ax.set_xlabel('Age')
	ax.set_ylabel('Hippocampus-to-event\ncorrelation')
	plt.rcParams['axes.xmargin'] = 0
	ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	print(HPC,', r = '+str(np.round(r2,2)),', p = '+str(np.round(p2,5)))
	fig.savefig(figurepath+'HPC/'+HPC+'_Age_vs_bump_size.png', bbox_inches='tight', dpi=300)

for HPC in ['HPC','aHPC','pHPC']:
	# plot all ages together
	sns.set_palette(colors_age)
	fig,ax = plt.subplots(1,1,figsize=(7,7))
	g = sns.lineplot(x='Time lag [s]', y='correlation_'+HPC, ax=ax, data=dfbumplag, ci=95,color=colors_age[3])
	ax.set_xlim([-10,10])
	ax.set_xlabel('Time (seconds)')
	ax.set_ylabel('Hippocampus-to-event\ncorrelation '+HPC)
	ax.margins(x=0)
	plt.savefig(figurepath+'HPC/'+HPC+'_bump_all.png', bbox_inches='tight',dpi=300)

for HPC in ['HPC','aHPC','pHPC']:
	sns.set(font_scale = 4,style="ticks")
	sns.set_palette(colors_age)
	# plot timecourse of xcorr
	fig,ax = plt.subplots(1,1,figsize=(10,12))
	g = sns.lineplot(x='Time lag [s]', y='correlation_'+HPC, hue='Age', ax=ax, data=dfbumplag, ci=95,linewidth = 5)
	#ax.set_xlim([-10,10])
	ax.set_xlabel('Time (seconds)')
	ax.set_ylabel('Hippocampus-to-event\ncorrelation')
	leg = ax.legend(loc='center', bbox_to_anchor=(0.5, -0.5))
	for line in leg.get_lines():
		line.set_linewidth(10)
	ax.margins(x=0)
	plt.savefig(figurepath+'HPC/'+HPC+'_bump_xcorr_ev_conv.png', bbox_inches='tight',dpi=300)
