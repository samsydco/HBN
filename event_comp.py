#!/usr/bin/env python3

# compare event timing from adult raters with event timing from children raters

import numpy as np
import pandas as pd
from settings import *

def get_boundaries(df,agedf):
	PartIDs = np.array(df.loc[df['Screen Name'] == 'Desc_Me']['Participant Public ID'].value_counts()[df.loc[df['Screen Name'] == 'Desc_Me']['Participant Public ID'].value_counts()>1].index)
	df=df[df['Participant Public ID'].isin(PartIDs)]
	agedf = agedf[agedf['Participant Public ID'].isin(PartIDs)]
	boundaries = pd.to_numeric(df.loc[df['Screen Name'] == 'Desc_Me']['Reaction Time']).values
	spike_boundaries = np.round(boundaries/1000/TR,0).astype(int)
	counts = np.append(np.bincount(spike_boundaries)[:-2],np.bincount(spike_boundaries)[-1])
	ev_conv = np.convolve(counts,hrf)[:nTR]
	# Subject ages:
	Ages = []
	for sub in PartIDs:
		subdf = agedf[agedf['Participant Public ID'].isin([sub])]
		Ages.append(pd.to_numeric(subdf[subdf['Question Key']=='age-year']['Response'].values)[0] + pd.to_numeric(subdf[subdf['Question Key']=='age-month']['Response'].values)[0] / 12)
	return spike_boundaries,ev_conv,Ages,df,agedf

def xcorr(a,b):
	# This helped convince me I'm doing the right thing:
	# https://currents.soest.hawaii.edu/ocn_data_analysis/_static/SEM_EDOF.html
	a = (a - np.mean(a)) / (np.std(a))
	b = (b - np.mean(b)) / (np.std(b))
	c = np.correlate(a, b, 'full')/max(len(a),len(b))
	return c

segpath = codedr + 'HBN_fmriprep_code/video_segmentation/'
ev_figpath = figurepath+'event_annotations/'

nTR = 750
TR = 0.8

# HRF (from AFNI)
dt = np.arange(0, 15,TR)
p = 8.6
q = 0.547
hrf = np.power(dt / (p * q), p) * np.exp(p - dt / q)

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

Prolificdf = pd.read_csv('data_exp_68194-v4/data_exp_68194-v4_task-1t2b.csv')
Prolificagedf = pd.read_csv('data_exp_68194-v4/data_exp_68194-v4_questionnaire-xtqr.csv')
Pro_spike_boundaries,Pro_ev_conv,Pro_Ages,Pro_df,Pro_agedf = get_boundaries(Prolificdf,Prolificagedf)

df4 = pd.read_csv('data_exp_61650-v4/data_exp_61650-v4_task-yi9p.csv')
agedf4 = pd.read_csv('data_exp_61650-v4/data_exp_61650-v4_questionnaire-pokv.csv')
df7 = pd.read_csv('data_exp_v7/data_exp_61650-v7_task-bycw.csv')
agedf7 = pd.read_csv('data_exp_v7/data_exp_61650-v7_questionnaire-vwly.csv')
df = pd.concat([df4, df7])
agedf = pd.concat([agedf4, agedf7])
child_spike_boundaries,child_ev_conv,child_Ages,child_df,child_agedf = get_boundaries(df,agedf)

# cross correlation between old-adults ratings and Prolific-adult ratings:
time = np.concatenate([np.arange(-nTR+1,0)*TR,np.arange(nTR)*TR])
idx = np.where(abs(time)<5)[0]
crosscorr = xcorr(ev_conv,Pro_ev_conv)
lag = time[idx[np.argmax(crosscorr[idx])]]

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	fig, (raw_ev_annot) = plt.subplots(figsize=(60, 20))
	a = raw_ev_annot.hist(ev_annot, bins=nTR, linewidth=20,color='b')
	b = raw_ev_annot.hist(child_spike_boundaries, bins=nTR, linewidth=20,color='r')
	c = raw_ev_annot.hist(Pro_spike_boundaries, bins=nTR, linewidth=20,color='g')


	fig, (hrf_ann) = plt.subplots(figsize=(60, 20))
	hrf_ann.plot(np.arange(nTR), ev_conv, linewidth=5,color='b')
	hrf_ann.plot(np.arange(nTR), child_ev_conv, linewidth=5,color='r')
	hrf_ann.plot(np.arange(nTR), Pro_ev_conv, linewidth=5,color='g')

	plt.plot(xcorrx[idx],crosscorr[idx])
	plt.xlabel('Time')
	plt.ylabel('Correlation')
