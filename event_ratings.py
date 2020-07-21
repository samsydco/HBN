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


nevent = []
ev_annot = []
for v in eventdict['timing'].values():
	ev_annot.extend(v)
	nevent.append(len(v))
ev_annot = np.asarray(ev_annot, dtype=int)

counts = np.append(np.bincount(ev_annot)[:-2],np.bincount(ev_annot)[-1])
ev_conv = np.convolve(counts,hrf,'same')

peaks = np.where(ev_conv>4)[0]

peakdiff = [0]+list(np.where(np.diff(peaks)>1)[0]+1)+[len(peaks)]

event_list = []
event_peak = []
for pi,pdi in enumerate(peakdiff[:-1]):
	seg = peaks[pdi:peakdiff[pi+1]]
	event_list.append(seg[np.argmax(ev_conv[seg])])
	event_peak.append(np.max(ev_conv[seg]))
evidx = np.argsort(event_peak)[-1:int(len(event_peak)-np.round(np.median(nevent)))-1:-1]
event_list = [e for ei,e in enumerate(event_list) if ei in evidx]
event_peak = [e for ei,e in enumerate(event_peak) if ei in evidx]

event_annot = [[]for e in event_list]
for ei,e in enumerate(event_list):
	for sub in eventdict['timing'].keys():
		for ti,t in enumerate(eventdict['timing'][sub]):
			if e-3<t and t<e+3:
				event_annot[ei].append(eventdict['annotation'][sub][ti])

if __name__ == "__main__":
	
	fig, (raw_ev_annot) = plt.subplots(figsize=(20, 20))
	counts = raw_ev_annot.hist(ev_annot, bins=nTR, alpha=0.25)


	xticks = list(np.arange(0, nTR, 50))
	yticks = list(np.arange(0, max(counts[0])+1, 2))
	plt.rcParams['xtick.labelsize']=30
	plt.rcParams['ytick.labelsize']=30

	raw_ev_annot.set_xticks(xticks)
	raw_ev_annot.set_yticks(yticks)
	raw_ev_annot.set_ylabel('# annotations', fontsize=40)
	raw_ev_annot.set_xlabel('Time in TRs', fontsize=40)
	raw_ev_annot.set_title('Event annotations', fontsize=40)
	plt.tight_layout()
	plt.savefig(ev_figpath+'ev_annots_hist.png', bbox_inches='tight')




	fig, (hrf_ann) = plt.subplots(figsize=(20, 20))
	hrf_ann.plot(np.arange(nTR), ev_conv, linewidth=5)
	plt.rcParams['xtick.labelsize']=30
	plt.rcParams['ytick.labelsize']=30
	hrf_ann.set_xlabel('TR (1 TR = 0.8 seconds)', fontsize=40)
	plt.savefig(ev_figpath+'hrf_conv.png', bbox_inches='tight')


	fig, (raw_ev_annot) = plt.subplots(figsize=(20, 20))
	counts = raw_ev_annot.hist(ev_annot, bins=nTR, alpha=0.25)
	raw_ev_annot.plot(np.arange(nTR), ev_conv, color='b',alpha=0.5,linewidth=5)
	raw_ev_annot.plot(event_list,event_peak,'b*',markersize=24)

	xticks = list(np.arange(0, nTR, 50))
	yticks = list(np.arange(0, max(counts[0])+1, 2))
	plt.rcParams['xtick.labelsize']=30
	plt.rcParams['ytick.labelsize']=30

	raw_ev_annot.set_xticks(xticks)
	raw_ev_annot.set_yticks(yticks)
	raw_ev_annot.set_ylabel('# annotations', fontsize=40)
	raw_ev_annot.set_xlabel('Time in TRs', fontsize=40)
	raw_ev_annot.set_title('Event annotations', fontsize=40)
	plt.savefig(ev_figpath+'ev_annots_both.png', bbox_inches='tight')
	
	
	# Compare annotations "ev_conv" with HPC bumps:
	# Precidence for thinking this way comes from Aya Ben-Yakov's event saliency measure
	from ISC_settings import *
	import seaborn as sns
	from scipy import stats
	from statsmodels.stats.multitest import multipletests
	colors_age = ['#edf8fb','#b3cde3','#8c96c6','#8856a7','#810f7c']
	xticks = [str(int(round(eqbins[i])))+\
		  ' - '+str(int(round(eqbins[i+1])))+' y.o.' for i in range(len(eqbins)-1)]
	xcorrx = np.concatenate([np.arange(-nTR+1,0)*TR,np.arange(nTR)*TR])
	D,ISC_w_time,ISC_g_time = dd.io.load(ISCpath+'HPC.h5',['/D','/ISC_w_time', '/ISC_g_time'])
	
	bumplagdict = {'Age':[],'correlation':[],'Time lag [s]':[],'Subj':[],'Exact Age':[]}
	for b in range(nbinseq-1):
		for subj,bumps in D[b].items():
			xcorrt = xcorr(bumps,ev_conv)#np.correlate(ev_conv,bumps,"full")#
			bumplagdict['Subj'].extend([subj]*len(xcorrx))
			bumplagdict['Age'].extend([xticks[b]]*len(xcorrx))
			bumplagdict['Exact Age'].extend([Phenodf['Age'][Phenodf['EID'] == subj.split('/')[-1].split('.')[0].split('-')[1]].values[0]]*len(xcorrx))
			bumplagdict['correlation'].extend(xcorrt)
			bumplagdict['Time lag [s]'].extend(xcorrx)
	dfbumplag = pd.DataFrame(data=bumplagdict)
	dfbumplag = dfbumplag[abs(dfbumplag['Time lag [s]'])<20]
	
	# Which time points post-0 are significantly different from zero?
	dfpost = dfbumplag[dfbumplag['Time lag [s]']>=0]
	times = dfpost['Time lag [s]'].unique()[1:]
	tvals = np.zeros(len(times))
	pvals = np.zeros(len(times))
	for ti,tp in enumerate(times):
		tvals[ti],pvals[ti] = stats.ttest_rel(dfpost[dfpost['Time lag [s]']==tp]['correlation'], dfpost[dfpost['Time lag [s]']==0.]['correlation'])
	pvals = pvals*len(pvals) # Bonferroni correction
	best_t = times[np.argmin(pvals)]
	r,p = stats.pearsonr(dfpost[dfpost['Time lag [s]']==best_t]['Exact Age'],dfpost[dfpost['Time lag [s]']==best_t]['correlation'])
	fig,ax=plt.subplots()
	sns.set_style("darkgrid", {"axes.facecolor": ".9"})
	sns.regplot(x='Exact Age', y="correlation", data=dfpost[dfpost['Time lag [s]']==best_t]).set_title('Delay = '+str(best_t)+'s\nr = '+str(np.round(r,2))+', p = '+str(np.round(p,2)))
	fig.savefig(figurepath+'HPC/Age_vs_bump_xcorr_ev_conv')
	
	# plot timecourse of xcorr with *'s for significance
	sns.set(font_scale = 1)
	sns.set_palette(colors_age)
	fig,ax = plt.subplots(1,1,figsize=(5,5))
	g = sns.lineplot(x='Time lag [s]', y='correlation', hue='Age', ax=ax, data=dfbumplag, ci=95)
	ax.plot(times[pvals<0.05],[0.090]*len(times[pvals<0.05]),'k*',markersize=8)
	ax.legend(loc='center', bbox_to_anchor=(0.5, -0.3))
	plt.savefig(figurepath+'HPC/bump_xcorr_ev_conv.png', bbox_inches='tight')
		
	
	
	wlagdict = {'Age':[],'correlation':[],'Time lag [s]':[],'s':[]}
	for b in range(nbinseq-1):
		for s in range(ISC_w_time.shape[1]):
			xcorrt = xcorr(ev_conv,ISC_w_time[2,s,b])#np.correlate(ev_conv,ISC_w_time[2,s,b],"full")#
			wlagdict['Age'].extend([xticks[b]]*len(xcorrx))
			wlagdict['s'].extend([s]*len(xcorrx))
			wlagdict['correlation'].extend(xcorrt)
			wlagdict['Time lag [s]'].extend(xcorrx)
	dfelag = pd.DataFrame(data=wlagdict)
	sns.set_palette(colors_age)
	fig,ax = plt.subplots(1,1,figsize=(5,5))
	g = sns.lineplot(x='Time lag [s]', y='correlation',
                hue='Age', ax=ax, data=dfelag[abs(dfelag['Time lag [s]'])<20], ci='sd')
	ax.legend(loc='center', bbox_to_anchor=(0.5, -0.3))
	plt.savefig(figurepath+'HPC/ISC_xcorr_within_ev_conv.png', bbox_inches='tight')

	import itertools
	glagdict = {'Age Pair':[],'correlation':[],'Time lag [s]':[],'s':[]}
	for p in itertools.combinations(range(nbinseq),2):
		if 4 in p:
			for s in range(len(ISC_g_time['splithalf'].keys())):
				ISC = ISC_g_time['splithalf'][s][str(p[0])+'_'+str(p[1])]
				xcorrt = xcorr(ev_conv,ISC)#np.correlate(ISC,ISC,'full')#
				glagdict['Age Pair'].extend([xticks[p[0]]+' with '+xticks[p[1]]]*len(xcorrx))
				glagdict['s'].extend([s]*len(xcorrx))
				glagdict['correlation'].extend(xcorrt)
				glagdict['Time lag [s]'].extend(xcorrx)
	dfglag = pd.DataFrame(data=glagdict)
	fig,ax = plt.subplots(1,1,figsize=(5,5))
	g = sns.lineplot(x='Time lag [s]', y='correlation',
                hue='Age Pair', ax=ax, data=dfglag[abs(dfglag['Time lag [s]'])<20], ci='sd')
	ax.legend(loc='center', bbox_to_anchor=(0.5, -0.3))
	plt.savefig(figurepath+'HPC/ISC_xcorr_g_ev_conv.png', bbox_inches='tight')


	

