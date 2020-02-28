#!/usr/bin/env python3

# import variables from 3_fmriprep.py:
m = __import__('3_fmriprep')
try:
    attrlist = m.__all__
except AttributeError:
    attrlist = dir (m)
for attr in attrlist:
    globals()[attr] = getattr (m, attr)
	
output_space = 'MNI152NLin2009cAsym'
path_tmp = path if site =='Site-RU' else path+site+'/'

for chunk in tqdm(pchunk):
	pstr = ' '.join(chunk)
	for task in ['DM','TP']:
		date = str(datetime.datetime.now())[0:19].replace(' ','_')
		f = open("%sfmriprep_cmdoutput/%s_%s_%s.txt"%(path,task,date,pstr.replace(" ","_")), "w")
		command = ('singularity run --cleanenv \
		/data/Loci/fmriprep-1.5.6.simg \
		-B '+path_tmp+':/data \
		-B '+outputdr+':/out \
		-B '+tmpdr+':/work \
		/data /out \
		participant \
		--ignore=slicetiming \
		--use-syn-sdc \
		--fs-license-file=/data/fs_license.txt \
		--output-spaces '+output_space+' \
		-w /scratch \
		--participant-label '+pstr+' -t movie'+task).split()
		sp.call(command,\
				 stdin=sp.PIPE, stderr=sp.PIPE,\
				 universal_newlines=True,stdout=f)