#!/usr/bin/env python3

# perform ppca on subjects with >60 tests

import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from ppca import PPCA
from statsmodels.multivariate.pca import PCA
from settings import *
from ppca_settings import *

Phenodf = pd.read_csv(metaphenopath+'allphenoaccnt.csv')
# Identify age-specific exams and combine them:
duplstr = ['YFAS','CBCL','SRS']
IQT = ['WISC','WASI','KBIT']
OT = ['ASR','YSR']
dupl = []
for d in duplstr:
	dupl.append([l for l in Phenodf.columns if d+'_' in l])
dupl = dupl+[IQT,OT]
non_incl_cols = ['Anon','Subj','Visit','Days','EID','Start','Study','Site','Year','Season','AGE','SEX','Total','Sex','Age','START_DATE','Baseline','Date']
duplcols = {}
# Identify which items are common among age-specific tests:
for d in dupl:
	combo = ' and '.join(d)
	pp= [p for p in glob.glob(assesspath+'*.csv') if any(substring in p for substring in ['_'+dd for dd in d])]
	cols = []
	for i,p in enumerate(pp):
		cols.append(list(pd.read_csv(p)))
		cols[i] = [c.replace('C_','') if 'C_S' in c else c for c in ['_'.join(x).upper() for x in [[x for x in ii if x not in duplstr+['Pre']+IQT+OT] for ii in [c.split('_') for c in cols[i] if not any(x in c for x in non_incl_cols) and not any(char.isdigit() for char in c if not any(x in c for x in ['Score','SRS']))]]]]
	if len(d) == 2:
		Phenodf[combo] = Phenodf[d[0]].fillna(Phenodf[d[1]])
		duplcols[combo] = set(cols[0]).intersection(set(cols[1]))
	elif len(d) == 3:
		Phenodf[combo] = Phenodf[d[0]].fillna(Phenodf[d[1]]).fillna(Phenodf[d[2]])
		duplcols[combo] = set(cols[0]).intersection(set(cols[1])).intersection(set(cols[2])).union(set(['IQ_P']))
	Phenodf = Phenodf.drop(columns=d)

# Only keep tests with greater than 50% respondents, 
Phenodf = Phenodf.loc[:, Phenodf.isnull().mean() < 0.5]
# Only keep subjects who filled out more than 60 tests
Phenodf = Phenodf[Phenodf['Total']>60]

datadf = Phenodf[['sub','Anonymized ID','Age','Sex','EHQ_Total']].copy()
phenodata = [p for p in glob.glob(assesspath+'*.csv') if any(substring in p for substring in ['_'+ s for s in [s for s in sum([s.split() for s in list(Phenodf)],[]) if 'and'!=s]]) and not any(x in p for x in ['EEG','Basic'])]#,'BIA','PreInt_Demos_Fam','FFQ'])]#
for p in tqdm(phenodata):
	abbr = p.split('_')[3]
	andstr = [s for s in list(Phenodf) if 'and' in s and abbr in s]
	if any(x in p for x in ['DailyMeds','KSADS']):
		df = pd.read_csv(p,low_memory=False)
	else:
		df = pd.read_csv(p)
	df = dfmod(p,df,non_incl_cols)
	cols = [c for c in df.columns if not any(x in c for x in non_incl_cols)]
	df[cols] = df[cols][1:].astype(float)
	df = df.groupby('Anonymized ID').mean()
	df.reset_index(level=0, inplace=True)
	if len(andstr)==0:
		cols = list(set([col for col in df.columns if not any(x in col for x in non_incl_cols+['Score'])])-set(datadf))+['Anonymized ID']
		datadf = datadf.merge(df[cols], how='left')
	else:
		cols = ['Anonymized ID']
		mapping = {}
		for c in list(duplcols[andstr[0]]):
			col = [m for m in df.columns if c.lower() in m.lower() and 'ADHP' not in m]
			if len(col) == 1:
				cols.append(col[0])
				mapping[cols[-1]]=andstr[0]+'_'+c
		datadf = datadf.merge(df[cols], how='left')
		if bool(set(mapping.values()) & set(datadf.columns)):
			for i,v in enumerate(list(mapping.values())):
				datadf[v] = datadf[v].fillna(datadf[list(mapping.keys())[i]])
			datadf = datadf.drop(columns=list(mapping.keys()))
		else:
			datadf = datadf.rename(index=str, columns=mapping)
		
datadf.to_csv(metaphenopath+'data.csv', index=False)

no_bs = False
if no_bs == True:
	datadf = pd.read_csv(metaphenopath+'data_no_bs.csv')
else:
	datadf = pd.read_csv(metaphenopath+'data.csv')


# drop columns where all values are the same:
datadf = datadf.drop(columns = [d for d in list(datadf.iloc[:,2:]) if min(datadf[d])==max(datadf[d])])
# drop 50 most nan-containing columns:
datadf = datadf.drop(columns = datadf.isnull().sum().sort_values()[-50:].index.tolist())
# drop 10 most nan-containing subjects:
datadf = datadf.drop(datadf.isnull().sum(1).sort_values().index.tolist()[-10:],axis=0)
# drop columns where all values are the same:
datadf = datadf.drop(columns = [d for d in list(datadf.iloc[:,2:]) if min(datadf[d])==max(datadf[d])])
data = datadf.iloc[:,2:].values

d = 10
# missing pca method from: https://github.com/allentran/pca-magic
ppca = PPCA()
ppca.fit(data,d=d)
variance_explained = [0]+[p for p in ppca.var_exp]
components = ppca.data
model_params = ppca.C
component_mat = ppca.transform()

a = np.dot(np.where(np.isnan(data),0,data), model_params[:,0])
b = np.dot(np.where(np.isnan(data),0,data), model_params[:,1])

import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(a,b)

ncomp = len(data)-np.max(sum(np.isnan(data)))-10

# missing pca method from: https://www.statsmodels.org/dev/generated/statsmodels.multivariate.pca.PCA.html
pc = PCA(data, ncomp = d, method='svd', missing='fill-em', tol=5e-08, max_iter=1000, tol_em=5e-08, max_em_iter=100)

# what are the variables that most contribute to each component?
for i in range(4):
	
	print('pc = ',i)
	print('Method from: www.statsmodels.org/dev/generated/statsmodels.multivariate.pca.PCA.html')
	print('variance explained = ',round(pc.rsquare[i+1]-pc.rsquare[i],4))
	print('Largest negative weights:')
	sortedweights = list(np.sort(pc.loadings[:,i])[:10])
	weightindexes = np.array(list(datadf.iloc[:,2:]))[list(np.argsort(pc.loadings[:,i])[:10])]
	for ii in range(10):
		print(weightindexes[ii],' = ',round(sortedweights[ii],4))
	print('Largest positive weights:')
	sortedweights = list(np.sort(pc.loadings[:,i])[-10:])
	weightindexes = np.array(list(datadf.iloc[:,2:]))[list(np.argsort(pc.loadings[:,i])[-10:])]
	for ii in range(10):
		print(weightindexes[ii],' = ',round(sortedweights[ii],4))
	print('correlation of pc with age = ',round(np.corrcoef(data[:,0],pc.factors[:,i])[0,1],4),'\n')
	print('Method from: https://github.com/allentran/pca-magic ')
	print('variance explained = ',round(variance_explained[i+1]-variance_explained[i],4))
	print('Largest negative weights:')
	sortedweights = list(np.sort(model_params[:,i])[:10])
	weightindexes = np.array(list(datadf.iloc[:,2:]))[list(np.argsort(model_params[:,i])[:10])]
	for ii in range(10):
		print(weightindexes[ii],' = ',round(sortedweights[ii],4))
	print('Largest positive weights:')
	sortedweights = list(np.sort(model_params[:,i])[-10:])
	weightindexes = np.array(list(datadf.iloc[:,2:]))[list(np.argsort(model_params[:,i])[-10:])]
	for ii in range(10):
		print(weightindexes[ii],' = ',round(sortedweights[ii],4))
	print('correlation of pc with age = ',round(np.corrcoef(data[:,0],component_mat[:,i])[0,1],4),'\n')
	
dd.io.save(metaphenopath+'data.h5',{'data':datadf,'ppca':ppca,'pc':pc})
	
subord = [sub.split('/')[-1].split('.h5')[0] for sub in subord]
sub
qsub = []
for sub in subord:
	sub = sub.split('/')[-1].split('.h5')[0]
	if datadf[datadf['sub'].str.match(sub)].empty:
		qsub.append(sub)

