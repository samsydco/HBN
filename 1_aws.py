#!/usr/bin/env python3

import subprocess as sp 
import numpy as np
import os
import csv
import glob
import fileinput
from settings import *

ps = sp.check_output(["aws","s3","ls","s3://fcp-indi/data/Archives/HBN/MRI/Site-RU/","--no-sign-request"]\
	).splitlines()
ps = [x.decode("utf-8") for x in ps] # convert from bytes to strings
ps = [s for s in ps if "sub" in s]
ps = [s[31:] for s in ps] # cutting out everything except sub*.tar.gz
#ps = ps [0:5] # arbitrarily downloading 5 subj (Many will have missing data)

# If Missingcsv exists, import info, if not, create it
MissingDict = {}
if os.path.exists(Missingcsv):
	with open(Missingcsv,'r') as f:
		r = csv.reader(f)
		rows = [i for i in r]
	for i in rows[1:]:
		MissingDict[i[0]] = [ii=='True' for ii in i[1:]]
else:
	with open(Missingcsv,'w') as f:
		w = csv.writer(f)
		w.writerow(["Subject","T1","DM","TP","fmap_AP","fmap_PA","DM len","TP len"])
# Check if dim4 = 750 for DM, and 250 for TP
for sub in ps:
	sub_temp = sub[:-len('.tar.gz')] # cutting out .tar.gz part of file name
	# check if sub exists in dir or Missingcsv
	if (sub_temp not in [i.replace(path,"") for i in glob.glob(path + 'sub*')] and
	sub_temp not in list(MissingDict.keys())):
		sp.run(["aws","s3","cp","s3://fcp-indi/data/Archives/HBN/MRI/Site-RU/%s"%(sub),path,"--no-sign-request"])
		sp.run(["tar","xvzf",path + sub,"-C",path])
		sp.run(["rm","-r",path+sub]) # removing .tar.gz file 
		# check if directory has anatomy folder, both movies,
		# and fmap containing both AP and PA directions
		conds = [os.path.exists(path+sub_temp+'/anat/'+sub_temp+'_acq-HCP_T1w.nii.gz'),\
		os.path.exists(path+sub_temp+'/func/'+sub_temp+'_task-movieDM_bold.nii.gz'), \
		os.path.exists(path+sub_temp+'/func/'+sub_temp+'_task-movieTP_bold.nii.gz'),\
		os.path.exists(path+sub_temp+'/fmap/'+sub_temp+'_dir-AP_acq-fMRI_epi.json'), \
		os.path.exists(path+sub_temp+'/fmap/'+sub_temp+'_dir-PA_acq-fMRI_epi.json')]
        conds.extend([np.nan,np.nan])
        if conds[1]:
            conds[5] = 750 == int(sp.check_output(('fslhd '+path+sub_temp+'/func/'+sub_temp+'_task-movieDM_bold.nii.gz').split()).split(b'dim4',1)[1].split()[0].decode("utf-8"))
        if conds[2]:
            conds[6] = 250 == int(sp.check_output(('fslhd '+path+sub_temp+'/func/'+sub_temp+'_task-movieTP_bold.nii.gz').split()).split(b'dim4',1)[1].split()[0].decode("utf-8"))
		if any(elem is False for elem in conds):
			MissingDict[sub_temp] = conds
			with open(Missingcsv,'a') as f:
				w = csv.writer(f)
				w.writerow([sub_temp]+[str(i) for i in conds])
			sp.run(["rm","-r",path+sub_temp]) # removing folder
'''
import pandas as pd
df = pd.read_csv(Missingcsv)
nFalse = 0
for sub in glob.glob(path+'sub*'):
    sub_temp = sub[16:]
    conds = [True]*5
    conds.append(750 == int(sp.check_output(('fslhd '+path+sub_temp+'/func/'+sub_temp+'_task-movieDM_bold.nii.gz').split()).split(b'dim4',1)[1].split()[0].decode("utf-8")))
    conds.append(250 == int(sp.check_output(('fslhd '+path+sub_temp+'/func/'+sub_temp+'_task-movieTP_bold.nii.gz').split()).split(b'dim4',1)[1].split()[0].decode("utf-8")))
    if not all(i for i in conds):
        df.loc[df.shape[0]] = [sub_temp]+conds
        sp.run(('mv '+path+sub_temp+'/ '+path+'bad_subs/').split())
        nFalse = nFalse + 1
        print(sub_temp,NFalse,conds) 
df['DM len'] = df['DM len'].map({np.nan: np.nan, 1: True, 0: False})
df['TP len'] = df['TP len'].map({np.nan: np.nan, 1: True, 0: False})
df.to_csv(Missingcsv, index=False)
'''         

# Need to add fMRI file names to "IntendedFor" field of fmap json's
# Remove irrelevent BOLD files
for sub in glob.glob(path+'sub*'):
	replacement_text = '[\"'+'\",\"'.join([t.replace(sub+'/','') for t in glob.glob(sub+'/func/*.nii.gz')])+'\"]'
	for f in glob.glob(sub+'/fmap/'+'*fMRI_epi.json'):
		with fileinput.FileInput(f,inplace=True,backup='.bak') as file:
			if '\"fMRI\"' in open(f).read():
				for line in file:
					print(line.replace('\"fMRI\"',replacement_text),end='')
			if '\"IntendedFor\":' not in open(f).read():
				for line in file:
					print(line.replace('\"PatientPosition\": \"HFS\"','\"PatientPosition\": \"HFS\",\n    \"IntendedFor\": '+replacement_text),end='')
                    
                    
 
