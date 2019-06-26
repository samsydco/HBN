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

goodsubs = [] # subjects in pipeline 
badsubs = []
collist = ['sub','goodbad','site']
df=pd.DataFrame(columns=collist) # all subjs with some MR data

for site in ['Site-RU','Site-CBIC']:
	print(site+':')
	# number available at Site-RU: 922
	print('number available: '+str(len(sp.check_output(["aws","s3","ls","s3://fcp-indi/data/Archives/HBN/MRI/"+site+"/","--no-sign-request"]\
	).splitlines()) - 2)) # subtract 'derivatives/' and 'participants.tsv'
	# number rejected due to missing data: 455
	awsdf = pd.read_csv(Missingcsv+'_'+site+'.csv',usecols=['Subject'])
	print('number rejected due to missing data: '+str(len(awsdf))) # from AWS QA
	badsubs = badsubs+[i.split('.')[0] for i in awsdf.Subject.tolist()] # subjects rejected (without '.')
	compdf = pd.read_csv(TRratingdr+'compT1_'+site+'.csv',usecols=['final','sub'])
	# number rejected due to bad T1: 146
	v = compdf['final'].value_counts()['n'] if site == 'Site-RU' else 0
	print('number rejected due to bad T1: '+str(sum(compdf['final'].isna()) + v))
	for index, row in compdf.iterrows():
		if (str(row['final']) != "n" and str(row['final'])!='nan'):
			goodsubs.append(row['sub'])
		else:
			badsubs.append(row['sub'])
	for key,s in {'good':goodsubs,'bad':badsubs}.items():
		df=df.append(pd.DataFrame([s,[key]*len(s),[site]*len(s)],index=collist).transpose(),ignore_index=True)

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

for key in ['good','bad']:
	gbhist = df.loc[df['goodbad'] == key]['Total'].hist(bins=20)
	gbhist.set_title(key+" Phenotypic Data Availability")
	gbhist.set_ylabel("Frequency")
	gbhist.set_xlabel("Number of tests available")
	plt.show()
	fig = gbhist.get_figure()
	fig.savefig(figurepath+key+"_Pheno_hist.png")

goodhist = df['Age'].hist(bins=21)
goodhist.set_title("Age range")
gbhist.set_ylabel("Frequency")
gbhist.set_xlabel("Age")
plt.show()

df['Total'][df['Total']>60].count()


