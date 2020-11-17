#!/usr/bin/env python3

# import variables from 3_fmriprep.py:
m = __import__('3_fmriprep')
try:
    attrlist = m.__all__
except AttributeError:
    attrlist = dir (m)
for attr in attrlist:
    globals()[attr] = getattr (m, attr)

path_tmp = path+site+'/'

for chunk in tqdm(pchunk):
	pstr = ' '.join([c[4:] for c in chunk])
	for task in ['DM']:
		date = str(datetime.datetime.now())[0:19].replace(' ','_')
		f = open("%sfmriprep_cmdoutput/%s_%s_%s.txt"%(path,task,date,pstr.replace(" ","_")), "w")
		command = ('singularity run --cleanenv \
		-B '+path_tmp+':/data \
		-B '+outputdr+':/out \
		-B '+tmpdr+':/work \
		'+path+codedr+'HBN_fmriprep_code/fmriprep-1.1.4.simg \
		/data /out \
		participant \
		--ignore=slicetiming \
		--fs-no-reconall \
		--fs-license-file=/data/fs_license.txt \
		-w /work/ \
		--participant-label '+pstr+' -t movie'+task).split()
		p = sp.Popen(command,\
				 stdin=sp.PIPE, stderr=sp.PIPE,\
				 universal_newlines=True,stdout=f)
		p.communicate('/n')[1]