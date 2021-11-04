#!/usr/bin/env python3

# compare event timing from adult raters with event timing from children raters

import numpy as np
import pandas as pd
from event_ratings import ev_annot, nTR, TR, hrf

adult_ev,_ = np.histogram(ev_annot, bins=nTR)
counts = np.append(np.bincount(ev_annot)[:-2],np.bincount(ev_annot)[-1])
ev_conv = np.convolve(counts,hrf)[:nTR]

df4 = pd.read_csv('data_exp_61650-v4/data_exp_61650-v4_task-yi9p.csv')
agedf4 = pd.read_csv('data_exp_61650-v4/data_exp_61650-v4_questionnaire-pokv.csv')
df7 = pd.read_csv('data_exp_v7/data_exp_61650-v7_task-bycw.csv')
agedf7 = pd.read_csv('data_exp_v7/data_exp_61650-v7_questionnaire-vwly.csv')
df = pd.concat([df4, df7])
agedf = pd.concat([agedf4, agedf7])
#df = pd.read_csv('data_exp_58692-v11_task-fxqt (1).csv')
PartIDs = np.array(df.loc[df['Screen Name'] == 'Desc_Me']['Participant Public ID'].value_counts()[df.loc[df['Screen Name'] == 'Desc_Me']['Participant Public ID'].value_counts()>1].index)
df=df[df['Participant Public ID'].isin(PartIDs)]
agedf = agedf[agedf['Participant Public ID'].isin(PartIDs)]
boundaries = pd.to_numeric(df.loc[df['Screen Name'] == 'Desc_Me']['Reaction Time']).values
boundaries2 = np.round(boundaries/1000/TR,0).astype(int)
counts = np.append(np.bincount(boundaries2)[:-2],np.bincount(boundaries2)[-1])
ev_conv2 = np.convolve(counts,hrf)[:nTR]
child_ev,_ = np.histogram(boundaries2, bins=nTR)

# Subject ages:
Ages = []
for sub in PartIDs:
	subdf = agedf[agedf['Participant Public ID'].isin([sub])]
	Ages.append(pd.to_numeric(subdf[subdf['Question Key']=='age-year']['Response'].values)[0] + pd.to_numeric(subdf[subdf['Question Key']=='age-month']['Response'].values)[0] / 12)

import matplotlib.pyplot as plt
fig, (raw_ev_annot) = plt.subplots(figsize=(60, 20))
a = raw_ev_annot.hist(ev_annot, bins=nTR, linewidth=20,color='b')
b = raw_ev_annot.hist(boundaries2, bins=nTR, linewidth=20,color='r')


fig, (hrf_ann) = plt.subplots(figsize=(60, 20))
hrf_ann.plot(np.arange(nTR), ev_conv, linewidth=5,color='b')
hrf_ann.plot(np.arange(nTR), ev_conv2, linewidth=5,color='r')