#!/usr/bin/env python3

import subprocess as sp 
import os
import csv
import glob
import fileinput

path = '/data/HBN/test2/'
codedr = 'fmriprep_code/'
ps = sp.check_output(["aws","s3","ls","s3://fcp-indi/data/Archives/HBN/MRI/Site-RU/","--no-sign-request"]\
	).splitlines()
ps = [x.decode("utf-8") for x in ps] # convert from bytes to strings
ps = [s for s in ps if "sub" in s]
ps = [s[31:] for s in ps] # cutting out everything except sub*.tar.gz
ps = ps [0:100] # arbitrarily downloading 100 subj (Many will have missing data)

MissingDict = {}
for sub in ps:
	sp.run(["aws","s3","cp","s3://fcp-indi/data/Archives/HBN/MRI/Site-RU/%s"%(sub),path,"--no-sign-request"])
	sp.run(["tar","xvzf",path + sub,"-C",path])
	sp.run(["rm","-r",path+sub]) # removing .tar.gz file
	sub=sub[0:16] # cutting out .tar.gz part of file name
	# check if directory has anatomy folder, both movies,
	# and fmap containing both AP and PA directions
	conds = [os.path.exists(path+sub+'/anat/'+sub+'_acq-HCP_T1w.nii.gz'),\
	os.path.exists(path+sub+'/func/'+sub+'_task-movieDM_bold.nii.gz'), \
	os.path.exists(path+sub+'/func/'+sub+'_task-movieTP_bold.nii.gz'),\
	os.path.exists(path+sub+'/fmap/'+sub+'_dir-AP_acq-fMRI_epi.json'), \
	os.path.exists(path+sub+'/fmap/'+sub+'_dir-PA_acq-fMRI_epi.json')]
	if any(elem is False for elem in conds):
		MissingDict[sub] = conds
		sp.run(["rm","-r",path+sub]) # removing folder

with open(path + codedr + 'Missing.csv','w') as f:
	w = csv.writer(f)
	w.writerow(["Subject","T1","DM","TP","fmap_AP","fmap_PA"])
	for key, value in MissingDict.items():
		w.writerow([key]+[str(i) for i in value])

# Need to add fMRI file names to "IntendedFor" field of fmap json's
for sub in glob.glob(path+'sub*'):
	replacement_text = '[\"'+'\",\"'.join([t.replace(sub+'/','') for t in glob.glob(sub+'/func/*.nii.gz')])+'\"]'
	for f in glob.glob(sub+'/fmap/'+'*fMRI_epi.json'):
		with fileinput.FileInput(f,inplace=True,backup='.bak') as file:
			for line in file:
				print(line.replace('\"fMRI\"',replacement_text),end='')


