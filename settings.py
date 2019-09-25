#!/usr/bin/env python3

path = '/data/HBN/test2/'
codedr = 'fmriprep_code/'#HBN_fmriprep_code/'
TRratingdr = path + codedr +'T1_rating/'
outputdr = path + 'fmriprep_output/'
tmpdr = '/tmp/scohen5'
Missingcsv = path + codedr + 'Missing'
T1file = path + codedr + 'T1.csv'
fmripreppath = outputdr + 'fmriprep/'
prepath = path + 'Preprocessed/'
ISCpath = path + 'ISCh5/'
HMMpath = path + 'HMM/'
phenopath = path + 'pheno_data/'
assesspath = path + 'assessment_data/rel_6/'
metaphenopath = path + 'pheno_meta_data/'
videopath = path+codedr+'HBN_fmriprep_code/videos/'
figurepath = path+'Figures/'

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

