#!/usr/bin/env python3

import subprocess as sp 
import numpy as np
import os
import csv
import glob
import tqdm
import fileinput
from settings import *

while True:
    try:
        site = input('Which site do you wish to collect data from: \"Site-RU\" or \"Site-CBIC\"?')
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue

    if site not in ['Site-RU','Site-CBIC']:
        print("Sorry, your response must be \"Site-RU\" or \"Site-CBIC\".")
        continue
    else:
        #Answer is good.
        break

ps = sp.check_output(["aws","s3","ls","s3://fcp-indi/data/Archives/HBN/MRI/"+site+"/","--no-sign-request"]\
	).splitlines()
ps = [x.decode("utf-8") for x in ps] # convert from bytes to strings
ps = [s for s in ps if "sub" in s]
ps = [s[31:] for s in ps] # cutting out everything except sub*.tar.gz
#ps = ps [0:5] # arbitrarily downloading 5 subj (Many will have missing data)

# If Missingcsv exists, import info, if not, create it
MissingDict = {}
Missingcsv = Missingcsv+'_'+site+'.csv'
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
if site == 'Site-CBIC':
	path = path + site + '/'
for sub in ps:
	sub_temp = sub[:-len('.tar.gz')] # cutting out .tar.gz part of file name
	# check if sub exists in dir or Missingcsv
	if (sub_temp not in [i.replace(path,"") for i in glob.glob(path + 'sub*')] and
	sub_temp not in list(MissingDict.keys())):
		sp.run(["aws","s3","cp","s3://fcp-indi/data/Archives/HBN/MRI/"+site+"/%s"%(sub),path,"--no-sign-request"])
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
					
# New BIDS convention: (for when using singularity)
#sub*/fmap/sub*_dir-AP_acq-fMRI_epi* -> sub*/fmap/sub*_acq-fMRI_dir-AP_epi*
#sub*/fmap/sub*_dir-PA_acq-fMRI_epi* -> sub*/fmap/sub*_acq-fMRI_dir-PA_epi*
import shutil
path='/data/HBN/test2/'
for site in ['Site-RU','Site-CBIC']:
	path_tmp = path + site + '/'
	for sub in tqdm.tqdm(glob.glob(path_tmp+'sub*')):
		sub_short = sub.split('/')[-1]
		for phase in ['AP','PA']:
			for file in ['_epi.json','_epi.nii.gz']:
				f = shutil.copy(sub+'/fmap/'+sub_short+'_dir-'+phase+'_acq-fMRI'+file, \
							    sub+'/fmap/'+sub_short+'_acq-fMRI_dir-'+phase+file)
				
# Add 'IntendedFor' to more subs' fmap jsons:
path='/data/HBN/test2/'
for site in ['Site-RU','Site-CBIC']:
	path_tmp = path + site + '/'
	for sub in tqdm.tqdm(glob.glob(path_tmp+'sub*')):
		replacement_text = '[\"'+'\",\"'.join([t.replace(sub+'/','') for t in glob.glob(sub+'/func/*.nii.gz')])+'\"]'
		for f in glob.glob(sub+'/fmap/'+'*.json'):
			with fileinput.FileInput(f,inplace=True,backup='.bak') as file:
				if '\"IntendedFor\":' not in open(f).read():
					for line in file:
						print(line.replace('\"PatientPosition\": \"HFS\"','\"PatientPosition\": \"HFS\",\n    \"IntendedFor\": '+replacement_text),end='')
	

                
 
