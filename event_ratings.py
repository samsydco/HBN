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
ev_conv = np.convolve(counts,hrf)[:nTR]

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
	
	time = np.arange(0,nTR)
	fig, (raw_ev_annot) = plt.subplots(figsize=(60, 20))
	counts = raw_ev_annot.hist(ev_annot, bins=nTR, linewidth=20,color='k')#,alpha=0.25)

	yticks = list(np.arange(0, max(counts[0])+1, 2))
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




	fig, (hrf_ann) = plt.subplots(figsize=(60, 20))
	hrf_ann.plot(np.arange(nTR), ev_conv, linewidth=5,color='k')
	hrf_ann.spines['right'].set_color('none')
	hrf_ann.spines['top'].set_color('none')
	plt.rcParams['xtick.labelsize']=80
	plt.rcParams['ytick.labelsize']=80
	hrf_ann.set_xlabel('TR (1 TR = 0.8 seconds)', fontsize=40)
	plt.savefig(ev_figpath+'hrf_conv.png', bbox_inches='tight')
	
	from ISC_settings import *
	D_ = dd.io.load(ISCpath+'HPC.h5',['/D'])[0]
	D = np.zeros(nTR)
	nsubj = 0
	for b in range(nbinseq):
		for bumps in D_[b].values():
			D=D+bumps
			nsubj+=1
	D=D/nsubj


	fig, (raw_ev_annot) = plt.subplots(figsize=(60, 20))
	counts = raw_ev_annot.hist(ev_annot, bins=nTR, linewidth=20,color='k')
	raw_ev_annot.plot(np.arange(nTR), ev_conv, color='k',alpha=0.5,linewidth=5)
	#raw_ev_annot.plot(event_list,event_peak,'b*',markersize=24)
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
	
	fig, (raw_ev_annot) = plt.subplots(figsize=(60, 20))
	counts = raw_ev_annot.hist(ev_annot, bins=nTR, linewidth=20,color='k')
	raw_ev_annot.plot(np.arange(nTR), ev_conv, color='k',alpha=0.5,linewidth=5)
	ax2 = raw_ev_annot.twinx()
	ax2.plot(np.arange(nTR)[5:], D[5:], color='k',linewidth=10)
	#raw_ev_annot.plot(event_list,event_peak,'b*',markersize=24)
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
	

			
	fig, ax = plt.subplots(figsize=(60, 20))
	ax.plot(np.arange(nTR)[5:], D[5:], color='k',linewidth=10)
	#raw_ev_annot.plot(event_list,event_peak,'b*',markersize=24)
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.set_xticks(np.append(time[0::nTR//5],time[-1]))
	ax.set_xticklabels([str(int(s*TR//60)) +':'+ str(int(s*TR%60))+'0' for s in time][0::nTR//5]+['10:00'], fontsize=80)
	#raw_ev_annot.set_yticks(yticks)
	plt.rcParams['xtick.labelsize']=50
	#plt.rcParams['ytick.labelsize']=80
	#raw_ev_annot.set_ylabel('# annotations', fontsize=80)
	ax.set_xlabel('Time (minutes)', fontsize=80)
	plt.tight_layout()
	plt.savefig(figurepath+'HPC/HPC_timecourse.png', bbox_inches='tight')
	
	
	# Compare annotations "ev_conv" with HPC bumps:
	# Precidence for thinking this way comes from Aya Ben-Yakov's event saliency measure
	from ISC_settings import *
	sizedict = dd.io.load(ISCpath+'HPC_vol.h5')
	sizedf = pd.DataFrame(data=sizedict)
	import statsmodels.api as sm
	import seaborn as sns
	from scipy import stats
	from statsmodels.stats.multitest import multipletests
	colors_age = ['#edf8fb','#b3cde3','#8c96c6','#8856a7','#810f7c']
	grey = 211/256
	xticks = [str(int(round(eqbins[i])))+\
		  ' - '+str(int(round(eqbins[i+1])))+' y.o.' for i in range(len(eqbins)-1)]
	xcorrx = np.concatenate([np.arange(-nTR+1,0)*TR,np.arange(nTR)*TR])
	for HPC in ['HPC','aHPC','pHPC']:
		D,ISC_w_time,ISC_g_time = dd.io.load(ISCpath+HPC+'.h5',['/D','/ISC_w_time', '/ISC_g_time'])
	
		bumplagdict = {'Age':[],'correlation':[],'Time lag [s]':[],'Subj':[],'Exact Age':[]}
		for b in range(nbinseq):
			for subj,bumps in D[b].items():
				if np.sum(np.isnan(bumps))!=nTR:
					xcorrt = xcorr(bumps,ev_conv)#counts)#np.correlate(ev_conv,bumps,"full")#
					bumplagdict['Subj'].extend([subj]*len(xcorrx))
					bumplagdict['Age'].extend([xticks[b]]*len(xcorrx))
					bumplagdict['Exact Age'].extend([Phenodf['Age'][Phenodf['EID'] == subj.split('/')[-1].split('.')[0].split('-')[1]].values[0]]*len(xcorrx))
					bumplagdict['correlation'].extend(xcorrt)
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
			tvals[ti],pvals[ti] = stats.ttest_1samp(dfpost[dfpost['Time lag [s]']==tp]['correlation'], 0)#stats.ttest_rel(dfpost[dfpost['Time lag [s]']==tp]['correlation'],dfpost[dfpost['Time lag [s]']==0.]['correlation'])
		pvals = pvals*len(pvals) # Bonferroni correction
		best_t_i = times[np.argmin(pvals)]
		best_t = 0 # Set from whole HPC
		tempdf = dfpost[dfpost['Time lag [s]']==best_t]
		tempsize = tempdf.merge(sizedf, on='Subj')
		r,p = stats.pearsonr(tempdf['Exact Age'],tempdf['correlation'])
		# no correlation between size and event-response!!
		r2,p2 = stats.pearsonr(tempsize[HPC],tempsize['correlation'])
		
		
		OLS_model = sm.OLS(tempsize['correlation'],tempsize[HPC]).fit()  # training the model
		residual_values = OLS_model.resid # residual values
		r,p = stats.pearsonr(tempsize['Exact Age'],tempsize['correlation'])
		
		X = sm.add_constant(tempsize[['Exact Age',HPC]]) # adding a constant
		model = sm.OLS(tempsize['correlation'], X).fit()
		predictions = model.predict(X)
		print_model = model.summary()
		print(print_model)
		
		sns.set(font_scale = 2,rc={'axes.facecolor':(grey,grey,grey)})
		fig,ax=plt.subplots(figsize=(7,5))
		sns.regplot(x='Exact Age', y="correlation", data=tempdf,color=colors_age[3])#.set_title('Delay = '+str(best_t)+'s\nr = '+str(np.round(r,2))+', p = '+str(np.round(p,2)))
		ax.set_xlabel('Age')
		ax.set_ylabel('Hippocampus-to-event\ncorrelation')
		plt.rcParams['axes.xmargin'] = 0
		print(HPC,', r = '+str(np.round(r,2)),', p = '+str(np.round(p,5)))
		fig.savefig(figurepath+'HPC/'+HPC+'_Age_vs_bump_xcorr_ev_conv.png', bbox_inches='tight', dpi=300)
		
		# plot indicating size
		fig,ax=plt.subplots(figsize=(7,5))
		sns.scatterplot(x='Exact Age', y="correlation", hue=HPC, size=HPC,
                linewidth=0,alpha=.5, 
                data=tempsize, ax=ax)
		#sns.relplot(x='Exact Age', y="correlation", hue=HPC, size=HPC,
		#			alpha=.5, color=colors_age[3],#palette="purple",
		#			data=tempsize)
		ax.set_xlabel('Age')
		ax.set_ylabel('Hippocampus-to-event\ncorrelation')
		plt.rcParams['axes.xmargin'] = 0
		ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		print(HPC,', r = '+str(np.round(r,2)),', p = '+str(np.round(p,5)))
		fig.savefig(figurepath+'HPC/'+HPC+'_Age_vs_bump_size.png', bbox_inches='tight', dpi=300)
		
		# plot all ages together
		sns.set(font_scale = 2,rc={'axes.facecolor':(grey,grey,grey)})
		sns.set_palette(colors_age)
		fig,ax = plt.subplots(1,1,figsize=(7,7))
		g = sns.lineplot(x='Time lag [s]', y='correlation', ax=ax, data=dfbumplag, ci=95,color=colors_age[3])
		ax.set_xlim([-10,10])
		ax.set_xlabel('Time (seconds)')
		ax.set_ylabel('Hippocampus-to-event\ncorrelation')
		ax.margins(x=0)
		plt.savefig(figurepath+'HPC/'+HPC+'_bump_all.png', bbox_inches='tight',dpi=300)
	
		# plot timecourse of xcorr with *'s for significance
		sns.set(font_scale = 2,rc={'axes.facecolor':(grey,grey,grey)})
		sns.set_palette(colors_age)
		fig,ax = plt.subplots(1,1,figsize=(7,7))
		g = sns.lineplot(x='Time lag [s]', y='correlation', hue='Age', ax=ax, data=dfbumplag, ci=95)
		#ax.plot(times[pvals<0.05],[0.090]*len(times[pvals<0.05]),'k*',markersize=15)
		ax.set_xlim([-10,10])
		#ax.set_xticks([-20,-10,-5,0,5,10,20])
		ax.set_xlabel('Time (seconds)')
		ax.set_ylabel('Hippocampus-to-event\ncorrelation')
		ax.legend(loc='center', bbox_to_anchor=(0.5, -0.5))
		ax.margins(x=0)
		plt.savefig(figurepath+'HPC/'+HPC+'_bump_xcorr_ev_conv.png', bbox_inches='tight',dpi=300)
		
	
	
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


	

