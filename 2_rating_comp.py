#!/usr/bin/env python3

import glob
import pandas as pd
import numpy as np
import os
import subprocess as sp 
import datetime
from settings import *

site = 'Site-CBIC'
#site = 'Site-RU'

GTdf = pd.read_csv(T1file, header=None)
compcsv = TRratingdr+'compT1_'+site+'.csv'
if os.path.exists(compcsv):
	compdf = pd.read_csv(compcsv)
else:
	if site == 'Site-RU':
		compdf = GTdf.copy()
		compdf.columns = ['sub','GT']
	else:
		compdf = pd.DataFrame(columns={'sub'})

if 'final' not in compdf.columns:
    compdf['final'] = [np.nan for _ in range(len(compdf))] 

for RAcsv in glob.glob(TRratingdr+'*'+site+'_T1.csv'):
    RA = RAcsv.replace(TRratingdr,"").replace('_'+site+'_T1.csv',"")
    if RA not in compdf.columns:
        compdf[RA] = [np.nan for _ in range(len(compdf))]
    RAdf = pd.read_csv(RAcsv, header=None)
    # check if subject in compdf, add if not
    for sub in RAdf.iloc[:,0]:
        yesno = RAdf[RAdf[0].str.match(sub)][1].iloc[0]
        if not compdf[compdf['sub'].str.match(sub)].empty:
            compdf.loc[np.where(compdf['sub'].str.match(sub).values)[0][0],RA] = yesno
        else:
            compdf = compdf.append(pd.Series(), ignore_index=True)
            compdf.loc[compdf.index[-1], 'sub'] = sub
            compdf.loc[compdf.index[-1], RA] = yesno
            
            
# make a tally:
if site == 'Site-RU':
	idx = [3,4,5]
else:
	idx = [2,3,4]
total = sum(all(str(ii) != 'nan' for ii in [i[ii] for ii in idx]) for i in compdf.values.tolist())

total_no_n = sum(all(ii != 'n' and str(ii) != 'nan' for ii in [i[ii] for ii in idx]) and not os.path.exists(fmripreppath+i[0]+'.html') for i in compdf.values.tolist())

total_ms = sum(all(str(ii) != 'nan' for ii in [i[ii] for ii in idx]) and sum(ii=='m' for ii in [i[ii] for ii in idx])>2 and not os.path.exists(fmripreppath+i[0]+'.html') for i in compdf.values.tolist())

total_yn = sum(all(str(ii) != 'nan' for ii in [i[ii] for ii in idx]) and 'n' in [i[ii] for ii in idx] and 'y' in [i[ii] for ii in idx] and not os.path.exists(fmripreppath+i[0]+'.html') for i in compdf.values.tolist())

total_potential = sum(all(str(ii) != 'nan' for ii in [i[ii] for ii in idx]) and not os.path.exists(fmripreppath+i[0]+'.html') for i in compdf.values.tolist())

print('total number of potential files: %s, total without no: %s, total with mostly maybe: %s, total with yes and no: %s'%(total_potential,total_no_n,total_ms,total_yn))

# look for ratings for which everyone has responded,
# check if there's a corresponding preprocessed file
# go from there...
for index, row in compdf.iterrows():
    sub = row['sub']
    yn = [str(j) for j in [row[e] for e in compdf.columns if e not in ('sub','final')]]
    if 'nan' not in yn[1:]: # does not include GT rating
        if 'n' not in yn:
            compdf.loc[index,'final'] = 'y'
        else:
            if yn.count('m')>2:
                compdf.loc[index,'final'] = 'y'
            if 'y' in yn:
                if str(row['final'])=='nan':
                    print(sub,yn)
                    sp.run(["fsleyes","%s%s/anat/%s_acq-HCP_T1w.nii.gz"%(path,sub,sub)])
                    yesno = yesnofun(sub)
                    if yesno in ['y','n','m']:
                        compdf.loc[index,'final'] = yesno
                    else:
                        break
# Need to save compdf to csv with 'final' column
compdf.to_csv(compcsv, index=False)


            
    

