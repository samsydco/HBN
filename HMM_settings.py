#!/usr/bin/env python3

# HMM Settings
from settings import *
from ISC_settings import *
from event_ratings import event_list

event_list = [56,206,244,343,373,404,443,506,544]
nevent = len(event_list)

tasks = ['DM','TP']
TR=0.8
nTR=[750,250]

TR1 = 12 #12 sec
TR2 = 300 #300 sec (5 min)
k_list = np.unique(np.round((10*60)/np.arange(TR1,TR2,TR1))).astype(int)
nsplit = 5
bins = [0,nbinseq-1]
win_range = np.arange(7,12) #range of windows tested for within - across corr
nshuff = 100

ROIopts = ['YeoROIsforSRM_sel_2020-01-14.h5','YeoROIsforSRM_2020-01-03.h5','SfN_2019/ROIs_Fig3/Fig3_','g_diff/']

# For 6_SRM.py, 7_HMM.py, and 7_HMM_timing.py
def makeROIdict(ROIfold):
	ROIs = {}
	for f in glob.glob(ROIfold):
		if len(glob.glob(ROIfold))==1:
			try:
				fs = dd.io.load(f,'/ROIs')
			except:
				fs = dd.io.load(f)
			for hemi, rs in fs.items():
				for r,subr in rs.items():
					f1 = hemi+'_'+str(round(float(r)))
					if type(subr) is list:
						ROIs[f1] = {}
						ROIs[f1]['vall'] = subr
						ROIs[f1]['nvox'] = len(subr)
						ROIs[f1]['hemi'] = hemi[0].capitalize()
						ROIs[f1]['tasks'] = ['DM','TP']
					else:
						for f2,subr_ in subr.items():
							ROIs[f1+'_'+f2] = {}
							ROIs[f1+'_'+f2]['vall'] = subr_
							ROIs[f1+'_'+f2]['nvox'] = len(subr_)
							ROIs[f1+'_'+f2]['hemi'] = hemi[0].capitalize()
							ROIs[f1+'_'+f2]['tasks'] = ['DM','TP']
		else:
			for f in glob.glob(ROIfold+'*roi'):
				fn = f.split(ROIfold)[1]
				roin = fn[:-7]
				ROI[roin] = {}
				if roin not in ROIs.keys():
					ROIs[f1]['tasks'] = 'DM' if fn[:2] == 'TP' else 'TP'
					ROIs[roin]['hemi'] = fn[3]
					vall = []
					with open(f, 'r') as inputfile:
						for line in inputfile:
							if len(line.split(' ')) == 3:
								vall.append(int(line.split(' ')[1]))
					ROIs[roin]['nvox'] = len(vall)
					ROIs[roin]['vall'] = vall
	return ROIs