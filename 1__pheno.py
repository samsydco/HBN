#!/usr/bin/env python3

#phenotypic data assessment:
# 1) How many subjects made it past QA check (for fMRI QA)
# 2) How much missing data is there?
# 3) Of the missing data, 
# how much of it is from:
# - subjects that made it past fMRI QA vs 
# - subjects that didn't vs 
# - subjects not-yet-downloaded?

import glob
import subprocess as sp 
import pandas as pd
from settings import *

goodsubs = [] # subjects in pipeline 

# number available at Site-RU: 922
totaws = len(sp.check_output(["aws","s3","ls","s3://fcp-indi/data/Archives/HBN/MRI/Site-RU/","--no-sign-request"]\
	).splitlines())
# number rejected due to missing data: 449
awsdf = pd.read_csv(Missingcsv)
badaws = len(awsdf) # from AWS QA
badsubs = dfaws['Subject'].tolist() # subjects rejected

# number rejected due to bad T1: 121
compdf = pd.read_csv(TRratingdr+'compT1.csv')
badT1 = sum(compdf['final'].isna()) + compdf['final'].value_counts()['n'] - sum(compdf['DS'].isna())
for index, row in compdf.iterrows():
	if (str(row['final']) != "n" and str(row['final'])!='nan'):
		goodsubs.append(row['sub'])
	elif str(row['DS'])!='nan':
		badsubs.append(row['sub'])
		
phenodata = glob.glob(path+'assessment_data/*.csv')
for p in phenodata:
	df = pd.read_csv(p)
	








