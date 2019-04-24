#!/usr/bin/env python3

import pandas as pd
import numpy as np

# replace text in columns using dictionary set in settings.py
def replacetext(df,ColumnName,dictname):
	df[ColumnName][1:][df[ColumnName][1:].notnull()] = [{key.lower() if type(key) == str else key: value for key, value in dictname.items()}[item] for item in [item.lower() for item in df[ColumnName][1:][df[ColumnName][1:].notnull()]]] 
	return df

def adddict(df,colname,dictname,value):
	for k in [k for k in df[colname].value_counts().index.tolist() if not k.lower() in list({k.lower(): v for k, v in dictname.items()}.keys())]:
			dictname[k] = value
	return dictname

def dfmod(p,df,non_incl_cols):
	
	# Pheno Text to number dicitonaries:
	frame = {'Small': 1,'Medium': 2,'Large': 3 }
	ActivityLevel = {'Very Light': 1,'Light': 2, 'Moderate': 3, 'Heavy': 4, 'Exceptional': 5} 

	EduHxlist = ['tvmovies', 'videocomputergames', 'computerinternet', 'listenmusic', 'reading', 'crafts', 'instrument', 'imaginativeplay', 'sports', 'clubsorganizations', 'homework', 'shopping', 'dancing', 'friendsathome', 'friendsout']
	EduHxlist.remove('clubsorganizations')

	parentrel = {'Excellent': 1, 'excellent': 1, 'perfect': 1, 'great': 1, 'Great': 1, 'very good': 1, 'awesome': 1, 'good': 2, 'Good, but always tension': 2, 'Good': 2, 'Fair': 3, 'Poor': 3}

	country = {'usa':1,'united states':1,'us':1}

	Time = {'None': 1, 'Never': 1, 'No': 1, 'Does not': 1, 'not yet': 1, '1x/ every 6 months': 1, '1x/every 2 weeks':1,'Rarely or never': 1, 'No playdates': 1, 'Rarely': 1, 'Rarely or never (summertime)': 1, 'DK': 1, '0': 1, '0 hours': 1, '1x/month': 2, '2x/month': 2, 'Less than 1 hr/wk': 2, '2-3x/month': 2, 'Less than 1 hour/week': 2, 'Less than 1 hr/wk Recorder': 2, '1/wk': 2, '2/week': 2, '1-3 hrs/wk': 3, '1x/week': 3, '1-3 hrs/wk Plays drums': 3, '1-3 hours/week': 3, '2/week': 3, '1x/week': 3, '2x/week': 3, '3/week': 3, '4-7 hrs/wk': 4, '6/week': 4, '5/week': 4, '8 hrs/wk': 4, '<1hr/day': 4, 'Less than 1 hr/day': 4, 'f': 4, '5-8 hrs/wk': 4, '2-3x/week':4, '4-7 hrs/wk*': 4, '10hrs/wk': 4, 'does mostly in afterschool program': 4, '1-3 hrs/day*': 5, '1-3 hrs/day; piano': 5, '1-3 hours': 5, '1-3 hrs/day ': 5, '3 hrs/day': 5, '10-12hrs/wk': 5, '1-3 hrs/day *': 5, '1-3 hrs/day': 5, '6-8 hrs/wk': 5, '20/week': 5, '10 hrs/wk': 5, '3x/wk': 5, '3x/week': 5, '11 hrs/wk': 5, '8-10 hrs/wk': 5, '12 hrs/wk': 5, '10/week': 5, '25-30hrs/wk': 6, '1-6 hrs/day': 6, '3-5x/week': 6, '3-6 hrs/day': 6, '3-6 hrs/day*': 6, '4x/week': 6, '6x/week': 6, '5x/week': 6, '7x/week': 6, '4-5 hours': 6, 'More than 6 hrs/day': 6, 'constant': 6, '15 hrs/wk': 6,'6-8 hrs/day': 6,'Daily': 6}

	nodict = {'no':1,'no ':1}
	hand = {'Right': 1, 'Left': 2, 'Ambidexterous': 3}

	dx01ca = {'Neurodevelopmental Disorders':1,
 'No Diagnosis Given: Incomplete Eval':2,
 'No Diagnosis Given':2,
 'Anxiety Disorders':3,
 'Depressive Disorders':4,
 'Disruptive':5,
 'Trauma and Stressor Related Disorders':3,
 'Obsessive Compulsive and Related Disorders':3}
	dx01s = {'Attention-Deficit/Hyperactivity Disorder':1,
 'Autism Spectrum Disorder':2,
 'Specific Learning Disorder':3,
 'Communication Disorder':4}
	dx01 = {'ADHD-Combined Type':1,
 'ADHD-Inattentive Type':2,
 'No Diagnosis Given: Incomplete Eval':3,
 'No Diagnosis Given':3,
 'Autism Spectrum Disorder':4,}
	dx01co = {'F90.2':1,'F90.0':2,'No Diagnosis Given: Incomplete Eval':3,'No Diagnosis Given':3,'F84.0':4,'F81.0':5,'F41.1':6,'F90.8':7,'F90.1':8,'F40.10':9,'F80.9':10}
	dxlist = [dx01ca,dx01s,dx01,dx01co]

	_ddict = {'extremely low':1,'very low':2,'low':3,'low average':4,'below aver':4,'below avg':4,'below avg.':4,'average':5,'high average':6,'high':7,'above aver':7,'very high':8,'extremely high':9}	

	lang = {'english':1}

	rel = {'adeoptive sister':3, 'adopted brother':2, 'adopted mother':1, 'adoptive  mother':1, 'adoptive father':4, 'adoptive father ':4, 'adoptive mother':1, 'adoptive mother (paternal aunt)':1, 'adoptive sister':3, 'bio dad':4, 'bio mom':1, 'bio mom ':1, 'bio mother':1, 'biolgical mother':1, 'biological brother':2, 'biological father':4, 'biological father ':4, 'biological mom ':1, 'biological mother':1, 'biological mother ':1, 'brother':2, 'fathe':4, 'father':4, 'father ':4, 'father (reporter)':4, 'foster mother':1, 'half brother':2, 'half sister':3, 'mom':1, 'mother':1, 'mother ':1, 'mother (reporter)':1,'mother biological':1,'paternal aunt/adopted mother':1,'sister':3,'stepsister':3}

	replacedict = {'BIA':[['Activity_Level','Frame'],[ActivityLevel,frame]],'Demos_Fam':[['P1_RelQuality','P2_RelQual'],[parentrel,parentrel]],'EduHx':[EduHxlist,[Time]*len(EduHxlist)],'Peg':[['peg_dom_hand'],[hand]],'CTOPP':[['CTOPP_EL_D','CTOPP_BW_D','CTOPP_NR_D','CTOPP_RD_D','CTOPP_RL_D','CTOPP_RSN_D'],[_ddict]*6]}

	langlist = []
	for c in ['Home_Lang1','Child_Lang_Understood','Child_Primary_Lang','Child_Lang1','Child_Lang1_Life','P1_Lang1','P1_Lang1_Life','P2_Lang1','P2_Lang1_Life','Sib_Lang1']:
		langlist.append([c,lang,2])

	adict = {'EduHx':[['clubsorganizations',Time,4]],'Demos_Fam':[['P1_CountryOrigin',country,2],['P2_CountryOrigin',country,2],['Child_CountryOrigin',country,2]],'TxHx':[['glasses',nodict,2],['hearingaid',nodict,2]],'Lang':langlist,'Home':[['fam_01_relation',rel,5],['fam_02_relation',rel,5],['fam_01_quality',parentrel,2],['fam_02_quality',parentrel,2]]}
	
	ppsmapping = {'PPS_F_01':'PPS_M_01','PPS_F_02':'PPS_M_02','PPS_F_03':'PPS_M_03','PPS_F_05':'PPS_M_06','PPS_F_Score':'PPS_M_Score'}
	
	if 'ConsensusDx' in p:
		df = df.drop(columns = [c for c in list(df) if 'Time' in c])
		for i,c in enumerate(['DX__Cat','DX__Sub','DX_','DX__Code']):
			for d in list(dxlist[i].keys()):
				df[d] = [np.nan for _ in range(len(df))] 
			for num in ['01','02','03','04','05','06','07','08','09','10']:
				cr = c[:3] + num + c[3:]
				for d in list(dxlist[i].keys()):
					df[d][df[cr].str.match(d).values == True] = 1
					df[d][df[cr].isin([k for k in df[cr].value_counts().index.tolist() if any(x in k for x in [dd for dd in list(dxlist[i].keys()) if dd!=d])]) & df[d].isna()] = 0
				df = df.drop(columns = cr)
	if 'PPS' in p:
		for g in ['M','F']:
			df['PPS_'+g+'_Score'] = df['PPS_'+g+'_Score'].iloc[1:].astype(float) / np.nanmax(df['PPS_'+g+'_Score'].iloc[1:].astype(float))
		for f in ppsmapping:
			df[ppsmapping[f]] = df[ppsmapping[f]].fillna(df[f])
		df = df.drop(columns=[c for c in list(df) if not any(l in c for l in list(ppsmapping.values())+['Anonymized ID'])])
	df = df.iloc[:,list(df.count()>max(df.count())/2)]
	if any(x in p for x in adict.keys()):
		ad = adict[[x for x in adict.keys() if x in p][0]]
		for i in ad:
			i[1] = adddict(df,i[0],i[1],i[2])
			df = replacetext(df,i[0],i[1])
	if 'KBIT' in p:
		df = df.drop(columns = [c for c in list(df) if 'Desc' in c])
	if 'Color' in p:
		df = df.drop(columns = ['CV_Plate_11','CV_Plate_14_R'])	
	if 'FamHx' in p:
		for c in [c for c in df.columns if not any(x in c for x in non_incl_cols)]:
			df[c][df[c].isin([k for k in df[c].value_counts().index.tolist() if 'no response' in k.lower()])] = np.nan
			df[c][df[c].isin([k for k in df[c].value_counts().index.tolist() if 'no ' in k.lower() or 'no' == k.lower() or 'none' == k.lower()])] = 1
			df[c][df[c].isin([k for k in df[c].value_counts().index.tolist() if type(k) is str and len(k)>2])] = 2	
	if 'EduHx' in p:
		df = df.drop(columns = ['religion'])
		for i in ['friendsathome','friendsout']:
			df[i][df[i].isin([k for k in df[i].value_counts().index.tolist() if not k in list(Time.keys())])] = np.nan
	if any(x in p for x in replacedict.keys()):
		replacecol = replacedict[[x for x in replacedict.keys() if x in p][0]]
		for i,r in enumerate(replacecol[0]):
			df = replacetext(df,r,replacecol[1][i])
	if 'Home' in p:
		for c in [c for c in list(df) if 'age' in c]:
			df[c][df[c].isin([k for k in df[c].value_counts().index.tolist() if any(x in k for x in ['+','m','^','?','y'])])] = np.nan
	if 'DevHx' in p:
		df['hospital_dur'][df['hospital_dur'].isin([k for k in df['hospital_dur'].value_counts().index.tolist() if any(x in k for x in ['F','d','?','+','*'])])] = np.nan
	return df



