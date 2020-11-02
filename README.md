# Code for analyses used in Cohen & Baldassano (2020). Title TBD.
## System specifications:
Python version 3.6.6
conda version 4.8.5
All code was run within a conda envirnoment specified in: environment.yml

You will also need to run:
pip install awscli --upgrade --user (This installs aws command line tools for your user)

The cortical results were made from fMRIPprep 1.1.4, installed in a docker container
The hippocampal results were made from fMRIprep 1.5.6, and run via a singularity image

## Order of code is as follows:
1) 1_aws.py: Download available HBN datasets from aws (from either Rutgers (RU) or CBCI (CitiGroup Cornell Brain Imaging Center)



