#!/usr/bin/env python3

path = '/data/HBN/test2/'
codedr = 'fmriprep_code/'#HBN_fmriprep_code/'
TRratingdr = path + codedr +'T1_rating/'
outputdr = path + 'fmriprep_output/'
tmpdr = path+'scratch/'
Missingcsv = path + codedr + 'Missing'
T1file = path + codedr + 'T1.csv'
fmripreppath = outputdr + 'fmriprep/'
fmripreppath_old = path + 'derivatives/'
prepath = path + 'Preprocessed/'
ISCpath = path + 'ISCh5/'
HMMpath = path + 'HMM/'
phenopath = path + 'pheno_data/'
assesspath = path + 'assessment_data/rel_6/'
metaphenopath = path + 'pheno_meta_data/'
videopath = path+codedr+'HBN_fmriprep_code/videos/'
figurepath = path+'Figures/'

# bad subjects for very unique reasons:
bad_sub_dict = {'sub-NDARHR140GMB':'TP BOLD scan is cut off, no motion params','sub-NDARYD546HCB':'TP BOLD scan is cut off, no motion params','sub-NDARJF517HC8':'*VNav_T1w.nii.gz damaged'}

def yesnofun(sub):
    while True:
        try:
            yesno = input("%s: Type \"y\" for \"yes\", \"n\" for \"no\", and \
                                \"m\" for \"maybe\".\n(If you need a break, type \"break\".)\n"%(sub))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

        if yesno not in ['y','n','m','break']:
            print("Sorry, your response must be \"y\", \"n\", or \"m\".")
            continue
        elif yesno=="break":
            break
        else:
            #Answer is good.
            break
    return yesno

import glob
subord = glob.glob(prepath+'sub*.h5')

