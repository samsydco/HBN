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

Prolificdf = pd.read_csv('data_exp_68194-v4/data_exp_68194-v4_task-1t2b.csv')
Prolificagedf = pd.read_csv('data_exp_68194-v4/data_exp_68194-v4_questionnaire-xtqr.csv')
Pro_spike_boundaries,Pro_ev_conv,Pro_Ages,Pro_df,Pro_agedf,Pro_agedict,Pro_gender,Pro_e_timing = get_boundaries(Prolificdf,Prolificagedf,[eqbins[-1],200])
# 14 F, 2 non-binary, 9 M
num_F_ = ['F' in g.upper() for g in Pro_gender]
num_B_ = ['B' in g.upper() for g in Pro_gender]

df4 = pd.read_csv('data_exp_61650-v4/data_exp_61650-v4_task-yi9p.csv')
agedf4 = pd.read_csv('data_exp_61650-v4/data_exp_61650-v4_questionnaire-pokv.csv')
df7 = pd.read_csv('data_exp_61650-v7/data_exp_61650-v7_task-bycw.csv')
agedf7 = pd.read_csv('data_exp_61650-v7/data_exp_61650-v7_questionnaire-vwly.csv')
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

# correlation between individual annotations:
orig_bounds = calc_indi_bounds(eventdict['timing'])
orig_df = pd.DataFrame(orig_bounds)
pro_df = pd.DataFrame(Pro_e_timing)
child_df = pd.DataFrame(child_e_timing)
old_df = pd.DataFrame(old_e_timing)
young_df = pd.DataFrame(young_e_timing)
df_dict = {'ori':orig_df,'pro':pro_df,'chi':child_df,'old':old_df,'you':young_df}

df_all = pd.concat([orig_df, pro_df, child_df],axis=1)
all_corr = np.array(df_all.corr())
all_corr_up = np.triu(all_corr,1)
corr_orig = all_corr_up[0:nsubj,0:nsubj]
corr_orig = corr_orig[np.nonzero(corr_orig)]
corr_pro = all_corr_up[nsubj:nsubj+len(Pro_Ages),nsubj:nsubj+len(Pro_Ages)]
corr_pro = corr_pro[np.nonzero(corr_pro)]
corr_chi = all_corr_up[nsubj+len(Pro_Ages):-1,nsubj+len(Pro_Ages):-1]
corr_chi = corr_chi[np.nonzero(corr_chi)]
corr_o_p = all_corr_up[0:nsubj,nsubj:nsubj+len(Pro_Ages)].flatten()
corr_o_c = all_corr_up[0:nsubj,nsubj+len(Pro_Ages):-1].flatten()
corr_p_c = all_corr_up[nsubj:nsubj+len(Pro_Ages),nsubj+len(Pro_Ages):-1].flatten()

if __name__ == "__main__":
	
	# Split-half between-group ISC
	# Pro-Young, Pro-Old, Young-Old, Orig-Pro, Orig-Kid
	import tqdm
	import itertools
	from ISC_settings import even_out,ISC_w_calc
	pairs = list(itertools.combinations(df_dict.keys(), 2))
	ISC_w = {k:np.zeros(nshuffle+1,2) for k in pairs}
	ISC_g = {k:np.zeros(nshuffle+1) for k in pairs}
	for p in pairs:
		ISC_g_time = np.zeros((nshuffle+1,1,nTR))
		ng1 = len(df_dict[p[0]].columns)
		ng2 = len(df_dict[p[1]].columns)
		n = np.min([ng1,ng2])
		if n%2 == 1: n -= 1
		for split in range(5):
			np.random.seed(split)
			df1 = np.array(df_dict[p[0]][np.random.choice(list(df_dict[p[0]].columns),n,replace=False)])
			df2 = np.array(df_dict[p[1]][np.random.choice(list(df_dict[p[1]].columns),n,replace=False)])
			D = np.expand_dims(np.concatenate([df1,df2],axis=1).T,axis=1)
			dim1 = np.concatenate([np.zeros(n),np.ones(n)])
			dim2 = np.concatenate([np.zeros(n//2),np.ones(n//2),np.zeros(n//2),np.ones(n//2)])
			subh = even_out(dim1,dim2)
			ISC_w[p][shuff], groups = ISC_w_calc(D,1,nTR,n,subh)
			ISC_b = []
			for htmp1 in [0,1]:
				for htmp2 in [0,1]:
					ISC_b.append(np.sum(np.multiply(groups[0,htmp1], groups[1,htmp2]), axis=1)/(nTR-1))
			denom = np.sqrt(ISC_w[p][shuff,0]) * np.sqrt(ISC_w[p][shuff,1])
			ISC_g[p][shuff] = np.sum(ISC_b, axis=0)/4/denom
			
	
	
	
	
	import tqdm
	nshuffle = 10000
	corr_list = ['ori','pro','chi','o_p','o_c','p_c']
	corr_lab = ['ori']*len(corr_orig) + \
			   ['pro']*len(corr_pro) + \
			   ['chi']*len(corr_chi) + \
			   ['o_p']*len(corr_o_p) + \
			   ['o_c']*len(corr_o_c) + \
			   ['p_c']*len(corr_p_c)
	corr_all = np.concatenate([corr_orig,corr_pro,corr_chi,corr_o_p,corr_o_c,corr_p_c])
	corr_dict = {k:np.zeros(nshuffle+1) for k in corr_list}
	for shuff in tqdm.tqdm(range(nshuffle+1)):
		for corr in corr_list:
			idx = [i for i,v in enumerate(corr_lab) if v==corr]
			corr_dict[corr][shuff] = np.mean(corr_all[idx])
		np.random.shuffle(corr_lab)
	p_dict = {k:[] for k in corr_list}
	for corr in corr_list:
		p_dict[corr] = np.sum(corr_dict[corr][0]<corr_dict[corr][1:])/nshuffle
	
	
	# Plot for Supplementary Figure 7:
	import matplotlib.pyplot as plt
	colors_age = ['#FCC3A1','#F08B63','#D02941','#70215D','#311638']

	fig, (hrf_ann) = plt.subplots(figsize=(12,4))
	hrf_ann.plot(np.arange(nTR)*TR, ev_conv/np.max(ev_conv), linewidth=1,color=colors_age[0])
	hrf_ann.plot(np.arange(nTR)*TR, Pro_ev_conv/np.max(Pro_ev_conv), linewidth=1,color=colors_age[2])
	hrf_ann.plot(np.arange(nTR)*TR, child_ev_conv/np.max(child_ev_conv), linewidth=1,color=colors_age[4])
	hrf_ann.legend(['In-lab (Adults)','Online (Adults)','Online (Children)'], fontsize=15)
	plt.xlim([0,600])
	plt.xlabel('Time [s]', fontsize=15)
	plt.ylabel('Boundary density', fontsize=15)
	
	
	import seaborn as sns
	sns.set_theme(style="whitegrid")

	data = pd.DataFrame(np.array([ev_conv/np.max(ev_conv), Pro_ev_conv/np.max(Pro_ev_conv), child_ev_conv/np.max(child_ev_conv)]).T, np.arange(nTR)*TR, columns=['In-lab (Adults)', 'Online (Adults)', 'Online (Children)'])

	sns.lineplot(data=data, palette="tab10", linewidth=2.5)

	plt.show()
