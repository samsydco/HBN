#!/usr/bin/env python3

# How atypical / typical is the HBN cohort studied compared to the general population in terms
# of the prevelancy of psychiatric disorders? 
# 1) How does the prevelency of clinical diagnoses in cohort studied compare to general population?
# 2) Is there a significant difference in diagnosed disorders based on age in cohort? 
# 3) If there is a significant difference in diagnoses between the ages, 
#    is this difference more significant than that in the general population?

# Disorders are stored in: assessment_data/rel_6/9994_ConsensusDx_20190329.xls
# Some disorder labels are in ppca_settings.py
# Previous treatment of phenotypic info is in 2__ppca.py

import glob
import pandas as pd
from ISC_settings import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def rate_plot(xval,xlab,ylab,name):
	order = [i for i in np.argsort(xval) if xval[i]>1]
	xval = [xval[i] for i in order]
	xlab =  [xlab[i]  for i in order]	
	x = np.arange(len(xval))
	fig, ax = plt.subplots(figsize=(11, 5))
	plt.bar(x, xval)
	plt.xticks(x, xlab)
	ax.set_ylabel(ylab)
	ax.set_xticklabels(xlab, rotation = 45, ha="right")
	plt.show()
	fig.savefig(figurepath+name, bbox_inches="tight")

p = assesspath+'9994_ConsensusDx_20190329.csv'
df = pd.read_csv(p)
# select subject IDs in df that also have fMRI data for:
df['EID']='sub-' + df['EID'].astype(str)
subs = [s.replace(prepath,'').replace('.h5','') for s in glob.glob(prepath+'sub*')]
df = df[df['EID'].isin(subs)]
# get ages and genders for subjects
Phenodf = pd.concat((pd.read_csv(f) for f in glob.glob(phenopath+'HBN_R*Pheno.csv')),ignore_index=True)
Phenodf['EID'] = 'sub-'+Phenodf['EID']
df = pd.merge(df,Phenodf[Phenodf["EID"].isin(list(df['EID']))].drop_duplicates(subset='EID')[['EID','Sex','Age']],on=['EID'])
eqbins = eqbins+[np.max(df.Age)]
xticks = [str(int(round(eqbins[i])))+\
		  ' - '+str(int(round(eqbins[i+1])))+' y.o.' for i in range(len(eqbins)-1)]
df['Age_Cat'] = pd.cut(df.Age,bins=eqbins,labels=xticks,include_lowest=True)
df['Sex'] = df['Sex'].astype(str).replace([str(0),str(1)],['Male','Female'])
# remove columns without diagnosis names in them:
nocollist = ['spec','code','time','confirmed','presum','rc','ruleout','past_doc','byhx','prem','rem','new','_ ','nodx','subject type','visit','days since enrollment','start_date','study','site','year','season']
clist = [c for c in list(df.columns) if any(ci in c.lower() for ci in nocollist)]
df.drop(clist, axis=1, inplace=True)
# list of columns with diagnoses in them:
clist = ['DX_'+str(c) if c>9 else 'DX_0'+str(c) for c in range(1,11)]
dummycolumns = [d for d in list(df.columns) if 'DX' in d and any(opt in d for opt in ['_Cat','_Sub']) or any(opt==d for opt in clist)]
dfd = pd.get_dummies(data=df, columns=dummycolumns)

# combine columns with same diagnoses:
cl2 = [c.replace('DX_01_','') for c in list(dfd.columns) if len(dfd[c].unique())==2 and not any(ci in c.lower() for ci in ['_new','_rem','_prem','past','_ ','_combined','level','consensusdx']) and 'DX_01' in c]
for c in cl2:
	clist = [ci for ci in list(dfd.columns) if c in ci]
	dfd[c] = dfd.loc[:,clist].any(axis=1)
	dfd.drop(clist, axis=1, inplace=True)
# another processing step to combine a few more columns:
for c in ['tic disorder','schizophrenia spectrum and other psychotic disorder','communication disorder']:
	clist = [c_ for c_ in list(dfd.columns) if c in c_.lower()]
	dfd[c] = dfd.loc[:,clist].any(axis=1)
	dfd.drop(clist, axis=1, inplace=True)
dfd.drop([c for c in list(dfd.columns) if '_ ' in c], axis=1, inplace=True)
dfd.to_csv(metaphenopath+'Neurodevelopmental_Diagnosis_Frequency.csv', index=False)

# plot percentage of each column
diagnosis = []
percentage = []
for c in list(dfd.columns)[2:]:
	diagnosis.append(c)
	percentage.append(dfd[c][dfd[c]==True].count())
percentage = [p/len(dfd) * 100 for p in percentage]

rate_plot(percentage,diagnosis,"Percentage Diagnosis",'diagnosis_freq.png')

# What are the ages/genders of these people?
bigdiag = [d for i,d in enumerate(diagnosis) if percentage[i]>1]
for i,dis in enumerate(bigdiag):
	g = sns.catplot(x="Age_Cat", hue="Sex", col=dis,data=dfd, kind="count")
	(g.set_axis_labels("Age Range", "Number of Subjects").set_xticklabels([x[:-5] for x in xticks]).set_titles("{col_name}").despine(left=True)).fig.suptitle(dis, size=16)
	g.fig.subplots_adjust(top=.85)
	g.savefig(figurepath+'Neurodevelopmental_Diagnosis_Frequency/'+dis.replace('/','_')+'.png', bbox_inches="tight")


# Why are there so many neurodevelopmental disorders? What are the sub-diagnoses of this catagory?
labels = []
counts = []
cat_cols = [c for c in list(df.columns) if 'Cat' in c]
for c in cat_cols:
	ccounts = df.loc[df[c] == 'Neurodevelopmental Disorders'][c[:6]+'Sub'].value_counts()
	for i in ccounts.index:
		if i not in labels:
			labels.append(i)
			counts.append(ccounts[i])
		else:
			counts[labels.index(i)] += ccounts[i]
counts = [c/sum(counts) * 100 for c in counts]
			
rate_plot(counts,labels,"Percentage of Neurodevelopmental Disorders",'neurdevel_freq.png')

# What are the prevelencies of each disorder within each age and sex group?


