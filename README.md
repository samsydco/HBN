# Code for analyses used in Cohen & Baldassano (2020). Title TBD.
## System specifications:
<p>Python version 3.6.6</p>
<p>conda version 4.8.5</p>
<p>All code was run within a conda envirnoment specified in: environment.yml</p>
<p>T1 scans were manually checked using FSL 5.0.11's fsleyes (see 2_RA_rating.py for details)</p>

<p>You will also need to run:</p>
<p>pip install awscli --upgrade --user (This installs aws command line tools for your user)</p>

<p>The cortical results were made from fMRIPprep 1.1.4, installed in a docker container</p>
<p>The hippocampal results were made from fMRIprep 1.5.6, and run via a singularity image</p>

## Settings files:
<ol>
<li> settings.py: 
<ul>
<li>sets various path</li>
<li>yesnofun: used in 2_RA_rating.py and 2_rating_comp.py</li>
</ul>
</li>
<li> ISC_settings.py: Creates functions used later in ISC code. 
<ul>
<li>make_phenol: creates demographic breakdown for all subjects (Age, Sex, phenotypic PC variables - not really used)</li>
<li>even_out: creates two groups with equal sizes and equal numbers of a demographic variable (e.g. sex)</li>
<li>bottom part of code creates 5 equally sized age groups of subjects</li>
</ul>
</li>
<li> HMM_settings.py: sets various variables used for HMM code:
<ul>
  <li>pandas dataframe with un-usable parcels is here (due to low log-likelihood in both age groups).</li>
  <li>FDR_p funciton is here</li>
</ul>
</li>
</ol>

## Order of code is as follows:
<ol>
<li> 1_aws.py: 
<ul>
  <li>Download available HBN datasets from aws (from either Rutgers (RU) or CBCI (CitiGroup Cornell Brain Imaging Center).</li>
  <li>Removed subjects who had files missing, and noted what was missing in csv file.</li>
  <li>Edited jsons for fieldmap scans for fmriprep compatibility</li>
  <li>Edited fieldmap scans names for fmriprep compatibility </li>
</ul>
</li>
<li> 2_RA_rating.py and 2_rating_comp.py:
<ul>
  <li>2_RA_rating.py: Allowed RA's to fill in ratings for T1 Scans</li>
  <li>2_rating_comp.py: Compare ratings between RAs: Compile RA ratings into one csv, Manually arbitrate between instances in which one RA said "yes" and another "no." </li>
</ul>
</li>
<li> 3_fmriprep.py and 3_run_singularity.py 
<ul>
  <li>3_fmriprep.py: Run fMRIPprep 1.1.4, installed in a docker container</li>
  <li>3_run_singularity.py: Run fMRIprep 1.5.6, via a singularity image</li>
</ul>
</li>
<li> 4_Preprocess.py and 4_Preprocess_HPC.py
<ul>
  <li>4_Preprocess.py: regresses out confounds found from fmriprep 1.1.4, and saves data in h5 file for cortex (one per subject).</li>
  <li>4_Preprocess_HPC.py: regresses out confounds found from fmriprep 1.5.6, labels anterior and posterior hippocampus, and saves data in h5 file for Hippocampus (one per subject).</li>
</ul>
</li>
<li> 5_parcellation.py: This creates h5 files for each parcel with the vertices contributing to that parcel and a matrix of size (subjects x vertices x time).
</li>
<li> Use vertices from 5_parcellation.py to calculate ISCs and HMMs in each age group: 6_ISC_sh_yeo.py and 6_HMM_n_k.py
<ul>
  <li>6_ISC.py: Determine difference in ISC and between-group ISC for Youngest and Oldest subjects.
    Do 100 subject-age permutations. (Repeatedly run 1000 more shuffles in parcels where p<0.05 or until p>0)</li>
  <li>6_HMM_n_k.py: Determine if there is a significant difference in the number of HMM-derived events between Youngest and Oldest groups.
    Do 100 subject-age permutaitons. (Repeatedly run 1000 more shuffles in parcels where p<0.05 or until p>0)</li>
</ul>
</li>
<li> 7_HMM_ll.py: 
<ul>
  <li>Find where HMM has a poor fit in either Youngest or Oldest subjects (HMM fit in 6_HMM_n_k).</li>
  <li>See if there is a difference in the number of events in the remaining parcels</li>
</ul>
</li>
<li> 8_HMM_stats.py: In remaining parcels:
<ul>
  <li>Run joint-fit HMM on Youngest and Oldest ages. Test on held-out subjects in all age groups</li>
  <li>Look for significant log-likelihood difference between Youngest and Oldest ages in jointly-fit HMMs.</li>
  <li>Look for significant "AUC" difference (HMM-prediction or event-timing) between Youngest and Oldest ages in jointly-fit HMMs.</li>
  <li>Null distribution is age-permuted subject-averaged timecourse.</li>
  <li>Run 100 permutations initially, then continue running code, adding 1000 permutations each time until p>0.05 or p!=0</li>
</ul>
</li>
<li> 9_p_check.py: Make an h5 file with p and q values for ISC and HMM tests. To be used for plotting on brain.
</li>
<li> Make Figures:
<ul>
  <li>10_HMM_AUC.py: Makes AUC plots (both with data and example plots).</li>
  <li>10_HMM_ll.py: Makes Log Likelihood plots in significant parcels.</li>
</ul>
</li>
</ol>

## Code Relevant for Hippocampus:
<ol>
<li>HPC.py: Gets average hippocampus trace for each subject (in both anterior and posterior hippocampus)</li>
<li>event_ratings.py: Calculates event timecourse from behavioral raters and compares to hippocampus timecourse.</li>
</ol>

## Other code:
<ul>
<li>6_ISC_test.py: Demonstrates that split-half ISC calculates ISC faster than Pairwise ISC or Leave-one-out ISC</li>
<li>1__pheno.py: Calculates various phenotypic information for sample such as how many subjects were eliminated at various pre-processing steps.</li>
<li>HMM_vs_hand.py: Compares HMM event timecourses to behavioral event timecourses</li>
</ul>


