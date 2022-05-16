#!/usr/bin/env python3

# compare event timing from adult raters with event timing from children raters

import numpy as np
import pandas as pd
import glob
from ISC_settings import eqbins
from scipy.stats import pearsonr

def get_boundaries(df,agedf,age_range):
	PartIDs = np.array(df.loc[df['Screen Name'] == 'Desc_Me']['Participant Public ID'].value_counts()[df.loc[df['Screen Name'] == 'Desc_Me']['Participant Public ID'].value_counts()>1].index)
	
    # Only use button pushes, not other lines like the end-of-movie log line
	row_mask = boundaries = (df['Screen Name'] == 'Desc_Me')&(df['Zone Type']=='response_keyboard_single')
	boundaries = pd.to_numeric(df.loc[row_mask]['Reaction Time']).values
	row_ids = df['Participant Public ID'][row_mask]
	
	# Eliminate subjects who click boundaries very rapidly
	# May indicate Bot or didn't understand task
	# Eliminate subjects who are outside of age_range
	# Eliminate subjects who identify only one event
	Bad_PartIDs = []
	for p in PartIDs:
		subdf = agedf[agedf['Participant Public ID'].isin([p])]
		Age = pd.to_numeric(subdf[subdf['Question Key']=='age-year']['Response'].values)[0] + pd.to_numeric(subdf[subdf['Question Key']=='age-month']['Response'].values)[0] / 12
		if np.median(np.diff(boundaries[row_ids == p])/1000)<1 or \
		Age < age_range[0] or Age > age_range[1] or \
		len(boundaries[row_ids == p])<2:
			Bad_PartIDs.append(p)
		
	PartIDs = PartIDs[np.invert(np.in1d(PartIDs, Bad_PartIDs))]
	
	df=df[df['Participant Public ID'].isin(PartIDs)]
	agedf = agedf[agedf['Participant Public ID'].isin(PartIDs)]

    # Bin to TRs first, then aggregate
    # This limits each participant to one button push per TR
	indi_boundaries = {}
	spike_boundaries = np.zeros(0, dtype=int)
	for p in PartIDs:
		indi_boundaries[p] = np.unique(np.round(boundaries[row_ids == p]/1000/TR,0).astype(int))
		spike_boundaries = np.append(spike_boundaries, indi_boundaries[p])
		
	indi_boundaries = calc_indi_bounds(indi_boundaries)	

	counts = np.bincount(spike_boundaries)
	ev_conv = np.convolve(counts,hrf)[:nTR]
	# Subject ages:
	Ages = []
	Genders = []
	Agedict = {}
	for sub in PartIDs:
		subdf = agedf[agedf['Participant Public ID'].isin([sub])]
		age = pd.to_numeric(subdf[subdf['Question Key']=='age-year']['Response'].values)[0] + pd.to_numeric(subdf[subdf['Question Key']=='age-month']['Response'].values)[0] / 12
		gender = subdf[subdf['Question Key']=='gender']['Response'].values[0]
		Ages.append(age)
		Genders.append(gender)
		Agedict[sub] = age
	
	return spike_boundaries,ev_conv,Ages,df,agedf,Agedict,Genders,indi_boundaries

def calc_indi_bounds(bound_dict):
	for k,v in bound_dict.items():
		bound_dict[k] = np.bincount(v)
		if len(bound_dict[k])<nTR:
			bound_dict[k] = np.concatenate([bound_dict[k], np.zeros(nTR - len(bound_dict[k]))])
		bound_dict[k] = np.convolve(bound_dict[k],hrf)[:nTR]
	return bound_dict

def lag_pearsonr(x, y, max_lags):
    """Compute lag correlation between x and y, up to max_lags
    Parameters
    ----------
    x : ndarray
        First array of values
    y : ndarray
        Second array of values
    max_lags: int
        Largest lag (must be less than half the length of shortest array)
    Returns
    -------
    ndarray
        Array of 1 + 2*max_lags lag correlations, for x left shifted by
        max_lags to x right shifted by max_lags
    """

    assert max_lags < min(len(x), len(y)) / 2, \
        "max_lags exceeds half the length of shortest array"

    assert len(x) == len(y), "array lengths are not equal"

    lag_corrs = np.full(1 + (max_lags * 2), np.nan)

    for i in range(max_lags + 1):

        # add correlations where x is shifted to the right
        lag_corrs[max_lags + i] = pearsonr(x[:len(x) - i], y[i:len(y)])[0]

        # add correlations where x is shifted to the left
        lag_corrs[max_lags - i] = pearsonr(x[i:len(x)], y[:len(y) - i])[0]

    return lag_corrs

nTR = 750
TR = 0.8

# HRF (from AFNI)
dt = np.arange(0, 15,TR)
p = 8.6
q = 0.547
hrf = np.power(dt / (p * q), p) * np.exp(p - dt / q)

eventdict = {key:{} for key in ['timing','annotation']}
for csv in glob.glob('video_segmentation/*csv'):
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

Prolificdf = pd.read_csv('Prolific_adult_data/event_data.csv')
Prolificagedf = pd.read_csv('Prolific_adult_data/age_data.csv')
Pro_spike_boundaries,Pro_ev_conv,Pro_Ages,Pro_df,Pro_agedf,Pro_agedict,Pro_gender,Pro_e_timing = get_boundaries(Prolificdf,Prolificagedf,[eqbins[-1],200])
# 14 F, 2 non-binary, 9 M
num_F_ = ['F' in g.upper() for g in Pro_gender]
num_B_ = ['B' in g.upper() for g in Pro_gender]

df4 = pd.read_csv('Children_data/Event_data_1.csv')
agedf4 = pd.read_csv('Children_data/age_data_1.csv')
df7 = pd.read_csv('Children_data/Event_data_2.csv')
agedf7 = pd.read_csv('Children_data/age_data_2.csv')
df = pd.concat([df4, df7])
agedf = pd.concat([agedf4, agedf7])
child_spike_boundaries,child_ev_conv,child_Ages,child_df,child_agedf,child_agedict,child_gender,child_e_timing = get_boundaries(df,agedf,[eqbins[0],eqbins[-1]])
# 22 F, 5 non-report, 39 M
num_F = ['F' in g.upper() or 'G' in g.upper() for g in child_gender]
num_Q = ['F' not in g.upper() and 'G' not in g.upper() and 'M' not in g.upper() for g in child_gender]

# Median split of children data to compare "Young" vs "Old" event timing:
median_age = np.median(child_Ages)
old_subjs = {k: v for k, v in child_agedict.items() if v > median_age}
young_subjs = {k: v for k, v in child_agedict.items() if v < median_age}
old_df      = child_df.loc   [child_df   ['Participant Public ID'].isin(old_subjs.keys())]
old_agedf   = child_agedf.loc[child_agedf['Participant Public ID'].isin(old_subjs.keys())]
young_df    = child_df.loc   [child_df   ['Participant Public ID'].isin(young_subjs.keys())]
young_agedf = child_agedf.loc[child_agedf['Participant Public ID'].isin(young_subjs.keys())]
old_child_spike_boundaries,old_child_ev_conv,old_child_Ages,old_child_df,old_child_agedf,old_child_agedict,old_gender,old_e_timing = get_boundaries(old_df,old_agedf,[eqbins[0],eqbins[-1]])
young_child_spike_boundaries,young_child_ev_conv,young_child_Ages,young_child_df,young_child_agedf,young_child_agedict,young_gender,young_e_timing = get_boundaries(young_df,young_agedf,[eqbins[0],eqbins[-1]])


if __name__ == "__main__":
	
	# individual annotation dataframes:
	orig_bounds = calc_indi_bounds(eventdict['timing'])
	orig_df = pd.DataFrame(orig_bounds)
	pro_df = pd.DataFrame(Pro_e_timing)
	child_df = pd.DataFrame(child_e_timing)
	old_df = pd.DataFrame(old_e_timing)
	young_df = pd.DataFrame(young_e_timing)
	df_dict = {'ori':orig_df,'pro':pro_df,'chi':child_df,'old':old_df,'you':young_df}
	
	# Split-half between-group ISC
	# Pro-Young, Pro-Old, Young-Old, Orig-Pro, Orig-Kid
	import tqdm
	import itertools
	from ISC_settings import even_out,ISC_w_calc
	nshuffle = 10000
	nsplit = 5
	maxlag = 10
	offsettimes = np.arange(-maxlag,maxlag+1)*TR
	pairs = list(itertools.combinations(df_dict.keys(), 2))
	ISC_g = {k:np.zeros((nsplit,nshuffle+1)) for k in pairs}
	lag_d = {k:np.zeros((nsplit,nshuffle+1)) for k in pairs}
	Ddict = {k:{k:[] for k in range(nsplit)} for k in pairs}
	gp_dict = {k:[] for k in pairs}
	lp_dict = {k:[] for k in pairs}
	for p in tqdm.tqdm(pairs):
		ng1 = len(df_dict[p[0]].columns)
		ng2 = len(df_dict[p[1]].columns)
		n = np.min([ng1,ng2])
		if n%2 == 1: n -= 1
		for s in range(nsplit):
			np.random.seed(s)
			df1 = np.array(df_dict[p[0]][np.random.choice(list(df_dict[p[0]].columns),n,replace=False)])
			df2 = np.array(df_dict[p[1]][np.random.choice(list(df_dict[p[1]].columns),n,replace=False)])
			Ddict[p][s] = np.expand_dims(np.concatenate([df1,df2],axis=1).T,axis=1)
			dim1 = np.concatenate([np.zeros(n),np.ones(n)])
			dim2 = np.concatenate([np.zeros(n//2),np.ones(n//2),np.zeros(n//2),np.ones(n//2)])
			for shuff in range(nshuffle+1):
				subh = even_out(dim1,dim2)
				ISC_w, groups = ISC_w_calc(Ddict[p][s],1,nTR,n,subh)
				ISC_b = []
				for htmp1 in [0,1]:
					for htmp2 in [0,1]:
						ISC_b.append(np.sum(np.multiply(groups[0,htmp1], groups[1,htmp2]), axis=1)/(nTR-1))
				denom = np.sqrt(ISC_w[0]) * np.sqrt(ISC_w[1])
				ISC_g[p][s,shuff] = np.sum(ISC_b, axis=0)/4/denom
				lag_d[p][s,shuff] = offsettimes[np.argmax( lag_pearsonr(np.mean(Ddict[p][s][dim1==1],0).T, np.mean(Ddict[p][s][dim1==0],0).T, maxlag))]
				np.random.shuffle(dim1)
		gp_dict[p] = np.sum(np.mean(ISC_g[p][:,0])>np.mean(ISC_g[p][:,1:],0))/nshuffle
		lp_dict[p] = np.sum(abs(np.mean(lag_d[p][:,0]))<abs(np.mean(lag_d[p][:,1:],0)))/nshuffle
		
	# Event bounds are where more than half of in-lab raters thought event occured (peaks)
	thresh = 0.5
	rel_ev = ev_conv/nsubj
	thresh_times = [i for i,ii in enumerate(rel_ev) if ii>thresh]
	diff = np.diff(thresh_times)
	event_times = [thresh_times[0]]+[thresh_times[i+1] for i,ii in enumerate(diff) if ii>1]
	peaks = []
	for e in event_times:
		peaks.append(e+np.argmax(rel_ev[e:e+10]))
		
	
	
	# Plot for "New" Figure 3:
	from settings import *
	ev_figpath = figurepath+'event_annotations/'
	import matplotlib.pyplot as plt
	from matplotlib import rcParams, rcParamsDefault
	rcParams.update(rcParamsDefault)
	colors_age = ['#FCC3A1','#F08B63','#D02941','#70215D','#311638']
	#A:
	fig, (hrf_ann) = plt.subplots(figsize=(12,4))
	hrf_ann.plot(np.arange(nTR)*TR, Pro_ev_conv/len(Pro_Ages), linewidth=2, color=colors_age[4])
	hrf_ann.plot(np.arange(nTR)*TR, old_child_ev_conv/len(old_child_Ages), linewidth=2, color=colors_age[2])
	hrf_ann.plot(np.arange(nTR)*TR, young_child_ev_conv/len(young_child_Ages), linewidth=2, color=colors_age[1])
	for p in peaks:
		hrf_ann.plot([p*TR,p*TR],[1.25,1.25],'k*')
		
	# Hide the right and top spines
	hrf_ann.spines['right'].set_visible(False)
	hrf_ann.spines['top'].set_visible(False)
	hrf_ann.legend(['Adults','Older Children','Younger Children'], fontsize=15)
	plt.xlim([0,600])
	plt.xlabel('Time (seconds)', fontsize=15)
	plt.ylabel('Boundary density', fontsize=15)
	plt.savefig(ev_figpath+'a_boundary_density.png', bbox_inches='tight',dpi=300)
	#B:
	lagdf = pd.DataFrame(columns = ['Time (seconds)', 'Boundary Correlation', 'Pair'])
	for p in [('pro', 'old'),('pro', 'you'),('old', 'you')]:
		if p == ('pro', 'old'): ptxt = 'Adults leading Older Children'
		elif p == ('pro', 'you'): ptxt = 'Adults leading Younger Children'
		elif p == ('old', 'you'): ptxt = 'Older Children leading Younger Children'
		for s in range(nsplit):
			n = Ddict[p][s].shape[0]//2
			dim1 = np.concatenate([np.zeros(n),np.ones(n)])
			corr = lag_pearsonr(np.mean(Ddict[p][s][dim1==1],0).T, np.mean(Ddict[p][s][dim1==0],0).T, maxlag)
			tempdf = pd.DataFrame({'Time (seconds)': offsettimes, 'Boundary Correlation': corr, 'Pair': [ptxt]*len(corr)})
			lagdf = lagdf.append(tempdf, ignore_index=True)
	
	import seaborn as sns
	sns.set(font_scale = 4,style="ticks")
	sns.set_palette([colors_age[4],colors_age[2],colors_age[1]])
	fig,ax = plt.subplots(1,1,figsize=(10,12))
	g = sns.lineplot(x='Time (seconds)', y='Boundary Correlation', hue='Pair', ax=ax, data=lagdf, ci=95,linewidth = 5)
	leg = ax.legend(loc='center', bbox_to_anchor=(0.5, -0.5))
	for line in leg.get_lines():
		line.set_linewidth(10)
	ax.margins(x=0)
	plt.savefig(ev_figpath+'b_boundary_correlation.png', bbox_inches='tight',dpi=600)
	
	# Note: in lead lag function, "leading" signal goes into second argument
	fig,ax = plt.subplots(1,1,figsize=(7,7))
	plt.plot(offsettimes,lag_pearsonr(old_child_ev_conv, Pro_ev_conv, maxlag),color=colors_age[4])
	plt.plot(offsettimes,lag_pearsonr(young_child_ev_conv, Pro_ev_conv, maxlag),color=colors_age[2])
	plt.plot(offsettimes,lag_pearsonr(young_child_ev_conv, old_child_ev_conv, maxlag),color=colors_age[1])
	plt.legend(['Adults leading Older Children', 'Adults leading Younger Children', 'Older Children leading Younger Children'])
	plt.plot([0,0],[0,1],'k--')
	plt.ylabel('Boundary Correlation', fontsize=15)
	plt.xlabel('Time (seconds)', fontsize=15)
	
	
	sns.set_theme(style="whitegrid")

	data = pd.DataFrame(np.array([ev_conv/np.max(ev_conv), Pro_ev_conv/np.max(Pro_ev_conv), child_ev_conv/np.max(child_ev_conv)]).T, np.arange(nTR)*TR, columns=['In-lab (Adults)', 'Online (Adults)', 'Online (Children)'])

	sns.lineplot(data=data, palette="tab10", linewidth=2.5)

	plt.show()
