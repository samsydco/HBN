#!/usr/bin/env python3

#phenotypic data assessment:
# 1) How many subjects made it past QA check (for fMRI QA)
# 2) How much missing data is there?
# 3) Of the missing data, 
# how much of it is from:
# - subjects that made it past fMRI QA vs 
# - subjects that didn't vs 
# - subjects not-yet-downloaded?

# what is the range of 

import glob
import subprocess as sp 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from settings import *

sites = ['Site-RU','Site-CBIC']
SID = 343 # From staten Island (no movies)
DR7 = 438 # Data release 7 is out!
subs = {key: {key: [] for key in ['goodsubs','badsubs']} for key in sites} # subjects in pipeline 
collist = ['sub','goodbad','site']
df=pd.DataFrame(columns=collist) # all subjs with some MR data
subtype = {key: [] for key in ['avail','missdata','badT1']}

for si,site in enumerate(sites):
	print(site+':')
	# number available at Site-RU: 922
	subtype['avail'].append(len(sp.check_output(["aws","s3","ls","s3://fcp-indi/data/Archives/HBN/MRI/"+site+"/","--no-sign-request"]\
	).splitlines()) - 2)
	print('number available: '+str(subtype['avail'][-1])) # subtract 'derivatives/' and 'participants.tsv'
	# number rejected due to missing data: 455
	awsdf = pd.read_csv(Missingcsv+'_'+site+'.csv',usecols=['Subject'])
	subtype['missdata'].append(len(awsdf))
	print('number rejected due to missing data: '+str(subtype['missdata'][-1])) # from AWS QA
	subs[site]['badsubs'].extend([i.split('.')[0] for i in awsdf.Subject.tolist()]) # subjects rejected (without '.')
	compdf = pd.read_csv(TRratingdr+'compT1_'+site+'.csv',usecols=['final','sub'])
	# number rejected due to bad T1: 146
	v = compdf['final'].value_counts()['n'] if site == 'Site-RU' else 0
	subtype['badT1'].append(sum(compdf['final'].isna()) + v)
	print('number rejected due to bad T1: '+str(subtype['badT1'][-1]))
	for index, row in compdf.iterrows():
		if (str(row['final']) != "n" and str(row['final'])!='nan'):
			subs[site]['goodsubs'].append(row['sub'])
		else:
			subs[site]['badsubs'].append(row['sub'])
	for key,s in subs[site].items():
		df=df.append(pd.DataFrame([s,[key]*len(s),
					[site]*len(s)],index=collist).transpose(),ignore_index=True)

Phenodf = pd.concat((pd.read_csv(f) for f in glob.glob(phenopath+'HBN_R*Pheno.csv')),ignore_index=True)
Phenodf = Phenodf.rename(index=str, columns={"EID": "sub"})	
Phenodf['sub'] = 'sub-'+Phenodf['sub']
df = pd.concat([df, Phenodf],sort=False).groupby('sub', as_index=False, sort=False).first()
df['Anonymized ID'] = [np.nan for _ in range(len(df))]

phenodata = glob.glob(assesspath+'*.csv')
#phenodata = phenodata[0:2]
for p in phenodata:
	abbr = p.split('_')[3]
	if any(abbr in s for s in [pi for pi in phenodata if pi != p]): # tests have multiple versions
		abbr = p.split('_')[3]+'_'+p.split('_')[4]	
	dftmp = pd.read_csv(p)	
	# sometimes EID is EID #1
	EID = 'EID #1' if 'DailyMeds' in p else 'EID'
	dftmp[EID]='sub-' + dftmp[EID].astype(str)
	subs = [i for i in dftmp[EID][1:] if type(i) != float]
	df[abbr] = [np.nan for _ in range(len(df))]
	if 'KSADS_2' not in abbr:
		# Add 'Anonymized ID' for KSADS
		df['Anonymized ID'] = df['Anonymized ID'].combine_first(df['sub'].map(dftmp.drop_duplicates(EID).set_index(EID)['Anonymized ID']))
		df.loc[[i for i, sub in enumerate(df['sub']) if sub in subs],abbr] = True
	else:
		df.loc[[i for i, sub in enumerate(df['Anonymized ID']) if sub in list(df['Anonymized ID'][1:])],abbr] = True

# How many items does everyone have?
# Females' 1.0 == True
df['Total'] = df.isin([True]).sum(1).subtract(df['Sex']) 

df.to_csv(metaphenopath+'allphenoaccnt.csv', index=False)

df = pd.read_csv(metaphenopath+'allphenoaccnt.csv')

%matplotlib inline

for key in ['goodsubs','badsubs']:
	gbhist = df.loc[df['goodbad'] == key]['Total'].hist(bins=20)
	gbhist.set_title(key+" Phenotypic Data Availability")
	gbhist.set_ylabel("Frequency")
	gbhist.set_xlabel("Number of tests available")
	plt.show()
	fig = gbhist.get_figure()
	fig.savefig(figurepath+key+"_Pheno_hist.png")

from ISC_settings import agel, eqbins
med = np.median(agel)
gooddf = df.loc[df['goodbad'] == 'goodsubs']
import seaborn as sns
plt.rcParams.update({'font.size': 15})
for i,s in enumerate(['Male','Female']):
	goodhist = sns.distplot(gooddf.loc[gooddf['Sex'] == i]['Age'],bins=eqbins, kde=False,label=s)
goodhist.legend()
goodhist.set_ylabel("Count")
#goodhist.axvline(x=med,color=[0.5,0.5,0.5],linestyle='--')
plt.tight_layout()
plt.show()
goodhist.get_figure().savefig(figurepath+'FLUX_2020/Age_hist.png')

df['Total'][df['Total']>60].count()


