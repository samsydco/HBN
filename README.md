# Code for analyses used in Cohen & Baldassano (2020). Title TBD.
## System specifications:
Python version 3.6.6
conda version 4.8.5
All code was run within a conda envirnoment specified in: environment.yml
T1 scans were manually checked using FSL 5.0.11's fsleyes (see 2_RA_rating.py for details)

You will also need to run:
pip install awscli --upgrade --user (This installs aws command line tools for your user)

The cortical results were made from fMRIPprep 1.1.4, installed in a docker container
The hippocampal results were made from fMRIprep 1.5.6, and run via a singularity image

## Settings files:
1) settings.py: sets various path

2) ISC_settings.py: Creates functions used later in ISC code. 
- make_phenol: creates demographic breakdown for all subjects (Age, Sex, phenotypic PC variables - not really used)
- even_out: creates two groups with equal sizes and equal numbers of a demographic variable (e.g. sex)
- bottom part of code creates 5 equally sized age groups of subjects

3) HMM_settings.py: sets various variables used for HMM code

## Order of code is as follows:
1) 1_aws.py: 
<ul>
  <li>Download available HBN datasets from aws (from either Rutgers (RU) or CBCI (CitiGroup Cornell Brain Imaging Center).</li>
  <li>Removed subjects who had files missing, and noted what was missing in csv file.</li>
  <li>Edited jsons for fieldmap scans for fmriprep compatibility</li>
  <li>Edited fieldmap scans names for fmriprep compatibility </li>
</ul>
2) 2_RA_rating.py: 
<ul>
<li>Allowed RA's to fill in ratings for T1 Scans</li>
  </ul>



