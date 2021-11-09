#!/usr/bin/env python3

# compare event timing from adult raters with event timing from children raters

import numpy as np
import pandas as pd
from event_ratings import ev_annot, nTR, TR, hrf

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
	
adult_ev,_ = np.histogram(ev_annot, bins=nTR)
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

import matplotlib.pyplot as plt
fig, (raw_ev_annot) = plt.subplots(figsize=(60, 20))
a = raw_ev_annot.hist(ev_annot, bins=nTR, linewidth=20,color='b')
b = raw_ev_annot.hist(child_spike_boundaries, bins=nTR, linewidth=20,color='r')
c = raw_ev_annot.hist(Pro_spike_boundaries, bins=nTR, linewidth=20,color='g')


fig, (hrf_ann) = plt.subplots(figsize=(60, 20))
hrf_ann.plot(np.arange(nTR), ev_conv, linewidth=5,color='b')
hrf_ann.plot(np.arange(nTR), child_ev_conv, linewidth=5,color='r')
hrf_ann.plot(np.arange(nTR), Pro_ev_conv, linewidth=5,color='g')