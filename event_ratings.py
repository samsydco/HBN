#!/usr/bin/env python3

import numpy as np
import pandas as pd
import deepdish as dd
import matplotlib.pyplot as plt
from settings import *

segpath = path + 'video_segmentation/'
ev_figpath = figurepath+'event_annotations/'

nTR = 750
TR = 0.8

# HRF (from AFNI)
dt = np.arange(0, 15,TR)
p = 8.6
q = 0.547
hrf = np.power(dt / (p * q), p) * np.exp(p - dt / q)

def xcorr(a,b):
	# This helped convince me I'm doing the right thing:
	# https://currents.soest.hawaii.edu/ocn_data_analysis/_static/SEM_EDOF.html
	a = (a - np.mean(a)) / (np.std(a))
	b = (b - np.mean(b)) / (np.std(b))
	c = np.correlate(a, b, 'full')/max(len(a),len(b))
	return c

eventdict = {key:{} for key in ['timing','annotation']}
for csv in glob.glob(segpath+'*csv'):
	initials = csv.split('/')[-1].split('-')[0]
	df = pd.read_csv(csv)
	if not any('TR' in c for c in df.columns):
		df.columns = df.iloc[0]
		df = df.iloc[1:]
		df = df.loc[:, df.columns.notnull()]
	TRstr = [t for t in df.columns if 'TR' in t][0]
	if TRstr != 'TR':
		df = df[(df['Scene Title '].notna()) & (df['Start TR'].notna())]
		df = df.rename(columns={'Scene Title ': 'Segment details'})
	eventdict['timing'][initials] = [int(tr) for tr in list(df[TRstr]) if not pd.isnull(tr)]
	eventdict['annotation'][initials] = list(df['Segment details'])

nsubj = len(eventdict['timing'])

sexl=['DA:F','SB:F','AR:F','FBM:M','TM:F','MT:F','MJ:F','TK:M','NR:M','CG:F','MN:M','EG:F','HC:F','HDZ:F', 'MB:M', 'SJ:F', 'MCK:F', 'FF:F', 'AS:M', 'DS:F', 'TC:F']

nevent = []
ev_annot = []
for v in eventdict['timing'].values():
	ev_annot.extend(v)
	nevent.append(len(v))
ev_annot = np.asarray(ev_annot, dtype=int)

counts = np.append(np.bincount(ev_annot)[:-2],np.bincount(ev_annot)[-1])
ev_conv = np.convolve(counts,hrf)[:nTR]

if __name__ == "__main__":
	
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
	
	from ISC_settings import *
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
	xcorrx = np.concatenate([np.arange(-nTR+1,0)*TR,np.arange(nTR)*TR])
	nsub = 415
	nsplit = 20
	y = [0]*int(np.floor(nsub/nsplit))*(nsplit-1)+[1]*int(np.floor(nsub/nsplit)+15)
	kf = KFold(n_splits=nsplit, shuffle=True, random_state=2)
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
				bumplagdict['correlation_'+HPC].extend(xcorr(Dall[HPC][b][subj],ev_conv))
			bumplagdict['Time lag [s]'].extend(xcorrx)
	dfbumplag = pd.DataFrame(data=bumplagdict)
	dfbumplag = dfbumplag[abs(dfbumplag['Time lag [s]'])<20]
	
	# Which time points post-0 are significantly different from zero?
	dfpost = dfbumplag[dfbumplag['Time lag [s]']>=0]
	dfpost=dfbumplag
	times = dfpost['Time lag [s]'].unique()#[1:]
	tvals = np.zeros(len(times))
	pvals = np.zeros(len(times))
	for ti,tp in enumerate(times):
		tvals[ti],pvals[ti] = stats.ttest_1samp(dfpost[dfpost['Time lag [s]']==tp]['correlation_HPC'],0)
	pvals = pvals*len(pvals) # Bonferroni correction
	best_t_i = times[np.argmin(pvals)]
	best_t = 0 # Set from whole HPC
	tempdf = dfpost[dfpost['Time lag [s]']==best_t]
	tempsize = tempdf.merge(sizedf, on='Subj')
	# test for linear vs polynomial fit of Age vs event_correlation:
	# both line and U fit equally well!
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
		g = sns.lineplot(x='Time lag [s]', y='correlation_'+HPC, hue='Age', ax=ax, data=dfbumplag, ci=95)
		#ax.plot(times[pvals<0.05],[0.090]*len(times[pvals<0.05]),'k*',markersize=15)
		ax.set_xlim([-10,10])
		#ax.set_xticks([-20,-10,-5,0,5,10,20])
		ax.set_xlabel('Time (seconds)')
		ax.set_ylabel('Hippocampus-to-event\ncorrelation')
		ax.legend(loc='center', bbox_to_anchor=(0.5, -0.5))
		ax.margins(x=0)
		plt.savefig(figurepath+'HPC/'+HPC+'_bump_xcorr_ev_conv.png', bbox_inches='tight',dpi=300)
