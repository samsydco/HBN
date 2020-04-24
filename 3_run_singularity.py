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

'''
plist = []
for f in glob.glob(outputdr+'/fmriprep/sub*/figures/'):
	if len(glob.glob(f+'*')) < 15:
		plist.append(f.split('/')[-3])

nchunk = 4 # number of items per chunk (maybe use 10?)
pchunk = [plist[x:x+nchunk] for x in range(0,len(plist),nchunk)]
# https://neurostars.org/t/using-use-syn-sdc-with-fieldmap-data/2592
# This works: --use-syn-sdc --ignore fieldmaps \
'''

for chunk in tqdm(pchunk):
	pstr = ' '.join([c[4:] for c in chunk])
	for task in ['DM','TP']:
		date = str(datetime.datetime.now())[0:19].replace(' ','_')
		f = open("%sfmriprep_cmdoutput/%s_%s_%s.txt"%(path,task,date,pstr.replace(" ","_")), "w")
		command = ('singularity run --cleanenv \
		-B '+path_tmp+':/data \
		-B '+outputdr+':/out \
		-B '+tmpdr+':/work \
		/data/Loci/fmriprep-1.5.6.simg \
		/data /out \
		participant \
		--ignore=slicetiming \
		--fs-license-file=/data/fs_license.txt \
		--output-spaces '+output_space+' \
		-w /work/ \
		--participant-label '+pstr+' -t movie'+task).split()
		p = sp.Popen(command,\
				 stdin=sp.PIPE, stderr=sp.PIPE,\
				 universal_newlines=True,stdout=f)
		p.communicate('/n')[1]