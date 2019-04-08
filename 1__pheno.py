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

# number available at Site-RU: 922
totaws = len(sp.check_output(["aws","s3","ls","s3://fcp-indi/data/Archives/HBN/MRI/Site-RU/","--no-sign-request"]\
	).splitlines()) - 2 # subtract 'derivatives/' and 'participants.tsv'
# number rejected due to missing data: 455
awsdf = pd.read_csv(Missingcsv+'_Site-RU.csv')
badaws = len(awsdf) # from AWS QA
badsubs = [i.split('.')[0] for i in awsdf['Subject'].tolist()] # subjects rejected (without '.')

# number rejected due to bad T1: 146
compdf = pd.read_csv(TRratingdr+'compT1.csv')
badT1 = sum(compdf['final'].isna()) + compdf['final'].value_counts()['n']
for index, row in compdf.iterrows():
	if (str(row['final']) != "n" and str(row['final'])!='nan'):
		goodsubs.append(row['sub'])
	else:
		badsubs.append(row['sub'])
		
global col_labs
col_labs = ['Anonymized ID','Age','Sex','EHQ_Total','Full_Pheno']
# Funciton for adding columns for data to baddf and gooddf
def add_cols(df,subs):
	df['sub'] = subs
	for col in col_labs:
		df[col] = [np.nan for _ in range(len(df))]
	return df
	
baddf = pd.DataFrame()
baddf = add_cols(baddf,badsubs)
gooddf = pd.DataFrame()
gooddf = add_cols(gooddf,goodsubs)

# funciton for adding demographic info (age,sex, etc.) to baddf and gooddf
def imp_demo(df,sub,row):
	if not df[df['sub'].str.match(sub)].empty:
		for col in col_labs[1:]:
			df.loc[np.where(df['sub'].str.match(sub).values)[0][0],col] = row[col]
	return df

Demodf = pd.concat((pd.read_csv(f) for f in glob.glob(phenopath+'HBN_R*Pheno.csv')),ignore_index=True)
for index, row in Demodf.iterrows():
	sub = 'sub-'+row['EID']
	baddf = imp_demo(baddf,sub,row)
	gooddf = imp_demo(gooddf,sub,row)

def add_pheno(gbdf,abbr,df,EID):
	gbdf[abbr] = [np.nan for _ in range(len(gbdf))]
	if 'KSADS_2' not in abbr:
		# Add 'Anonymized ID' for KSADS
		gbdf['Anonymized ID'] = gbdf['Anonymized ID'].combine_first(gbdf['sub'].map(df.drop_duplicates(EID).set_index(EID)['Anonymized ID']))
		gbdf.loc[[i for i, sub in enumerate(gbdf['sub']) if sub in subs],abbr] = True
	else:
		gbdf.loc[[i for i, sub in enumerate(gbdf['Anonymized ID']) if sub in list(df['Anonymized ID'][1:])],abbr] = True
	return gbdf

phenodata = glob.glob(path+'assessment_data/*.csv')
#phenodata = phenodata[0:2]
for p in phenodata:
	abbr = p.split('_')[2]
	if any(abbr in s for s in [pi for pi in phenodata if pi != p]): # tests have multiple versions
		abbr = p.split('_')[2]+'_'+p.split('_')[3]	
	df = pd.read_csv(p)	
	# sometimes EID is EID #1
	EID = 'EID #1' if 'DailyMeds' in p else 'EID'
	df[EID]='sub-' + df[EID].astype(str)
	subs = [i for i in df[EID][1:] if type(i) != float]
	baddf = add_pheno(baddf,abbr,df,EID)
	gooddf = add_pheno(gooddf,abbr,df,EID)

# How many items does everyone have?
# Females' 1.0 == True
gooddf['Total'] = gooddf.isin([True]).sum(1).subtract(gooddf['Sex']) 
baddf['Total'] = baddf.isin([True]).sum(1).subtract(baddf['Sex'])

gooddf.to_csv(metaphenopath+'good.csv', index=False)
baddf.to_csv(metaphenopath+'bad.csv', index=False)

gooddf = pd.read_csv(metaphenopath+'good.csv')
baddf = pd.read_csv(metaphenopath+'bad.csv')



%matplotlib inline
datadict = {'good':gooddf,'bad':baddf}

for key,df in datadict.items():
	gbhist = df['Total'].hist(bins=20)
	gbhist.set_title("Phenotypic Data Availability")
	gbhist.set_xlabel("Frequency")
	gbhist.set_ylabel("Number of tests available")
	plt.show()
	fig = gbhist.get_figure()
	fig.savefig(figurepath+key+"_Pheno_hist.png")

goodhist = df['Age'].hist(bins=21)
goodhist.set_title("Age range")
gbhist.set_xlabel("Frequency")
gbhist.set_ylabel("Age")
plt.show()



gooddf['Total'][gooddf['Total']>60].count()
gooddf['sub'][gooddf['Total']>60]



		









