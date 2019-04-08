#!/usr/bin/env python3

# perform ppca on subjects with >60 tests

import glob
import pandas as pd
from ppca import PPCA
from statsmodels.multivariate.pca import PCA
from settings import *

gooddf = pd.read_csv(metaphenopath+'good.csv')
dupl = [[l for l in gooddf.columns if 'YFAS_' in l],[l for l in gooddf.columns if 'CBCL_' in l],[l for l in gooddf.columns if 'SRS_' in l],['WISC','WASI','KBIT'],['ASR','YSR']]
for d in dupl:
	combo = ' and '.join(d)
	if len(d) == 2:
		gooddf[combo] = gooddf[d[0]].fillna(gooddf[d[1]])
	elif len(d) == 3:
		gooddf[combo] = gooddf[d[0]].fillna(gooddf[d[1]]).fillna(gooddf[d[2]])
	gooddf = gooddf.drop(columns=d)
	
gooddf = gooddf.loc[:, gooddf.isnull().mean() < 0.5]
gooddf = gooddf[gooddf['Total']>60]

datadf = gooddf[['sub','Anonymized ID','Age','Sex','EHQ_Total']].copy()
phenodata = [p for p in glob.glob(path+'assessment_data/*.csv') if any(substring in p for substring in [s for s in sum([s.split() for s in list(gooddf)],[]) if 'and'!=s])]
non_incl_cols = ['Anon','Subj','Visit','Days','EID','Start','Study','Site','Year','Season','Score','AGE','SEX','Total','Sex','Age','START_DATE']
for p in phenodata:
	abbr = p.split('_')[2]
	andstr = [s for s in list(gooddf) if 'and' in s and abbr in s]
	df = pd.read_csv(p)
	if len(andstr)==0:
		cols = [col for col in df.columns if not any(x in col for x in non_incl_cols)]+['Anonymized ID']
		datadf = datadf.merge(df[cols], how='left')
	else:
		cols = [col for col in df.columns if any(x in col for x in ['Score','Anonymized ID'])]
		mapping = {}
		for c in cols:
			mapping[c]=andstr[0]+'_'+c.split('_')[-1]
		del mapping['Anonymized ID']
		datadf = datadf.merge(df[cols], how='left')
		if bool(set(mapping.values()) & set(datadf.columns)):
			for i,v in enumerate(list(mapping.values())):
				datadf[v] = datadf[v].fillna(datadf[list(mapping.keys())[i]])
			datadf = datadf.drop(columns=list(mapping.keys()))
		else:
			datadf = datadf.rename(index=str, columns=mapping)
		
datadf.to_csv(metaphenopath+'data.csv', index=False)	
		


data = datadf.as_matrix(columns=None)


ppca = PPCA(data)