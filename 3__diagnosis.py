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

import pandas as pd


p = '/data/HBN/test2/assessment_data/rel_6/9994_ConsensusDx_20190329.csv'
df = pd.read_csv(p)
# list of columns with diagnoses in them:
clist = ['DX_'+str(c) if c>9 else 'DX_0'+str(c) for c in range(1,11)]
dummycolumns = [d for d in list(df.columns) if any(opt in d for opt in ['_Cat','_Sub']) or any(opt==d for opt in clist)]
dfd = pd.get_dummies(data=df, columns=dummycolumns)


cl2 = [c for c in list(dfd.columns) if len(dfd[c].unique())==2 and not any(ci in c.lower() for ci in ['_new','_rem','_prem','past','_ ','_combined','level','consensusdx']) and 'DX_01' in c]
dfd.loc[:,cl2].any(axis=1)



