#!/usr/bin/env python3


import numpy as np
import pandas as pd
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
	eventdict['timing'][initials] = list(df[TRstr])
	eventdict['annotation'][initials] = list(df['Segment details'])

nsubj = len(eventdict['timing'])

ev_annot = []
for v in eventdict['timing'].values():
	ev_annot.extend(v)
ev_annot = np.asarray(ev_annot)


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
plt.savefig(ev_figpath+'ev_annots_hist.png', bbox_inches='tight')

ev_conv = np.convolve(counts[0],hrf,'same')

peaks = np.where(ev_conv>4)[0]

peakdiff = [0]+list(np.where(np.diff(peaks)>1)[0]+1)+[len(peaks)]

event_list = []
for pi,pdi in enumerate(peakdiff[:-1]):
	seg = peaks[pdi:peakdiff[pi+1]]
	event_list.append(seg[np.argmax(ev_conv[seg])])


fig, (hrf_ann) = plt.subplots(figsize=(20, 20))
hrf_ann.plot(np.arange(nTR), ev_conv, linewidth=5)
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
hrf_ann.set_xlabel('TR (1 TR = 0.8 seconds)', fontsize=40)
plt.savefig(ev_figpath+'hrf_conv.png', bbox_inches='tight')


fig, (raw_ev_annot) = plt.subplots(figsize=(20, 20))
counts = raw_ev_annot.hist(ev_annot, bins=nTR, alpha=0.25)
raw_ev_annot.plot(np.arange(nTR), ev_conv, color='b',alpha=0.5,linewidth=5)
raw_ev_annot.plot(event_list,ev_conv[event_list],'b*',markersize=24)

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

