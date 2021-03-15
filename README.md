# Code for analyses used in Cohen & Baldassano (2020). Title TBD.
## System specifications:
<p>Python version 3.6.6</p>
<p>conda version 4.8.5</p>
<p>All code was run within a conda envirnoment specified in: <a href="https://github.com/samsydco/HBN/blob/master/environment.yml">environment.yml</a></p>
<p>T1 scans were manually checked using FSL 5.0.11's fsleyes (see 2_RA_rating.py for details)</p>

<p>You will also need to run:</p>
<pre style="color: silver; background: black;">pip install awscli --upgrade --user</pre>
<p>(This installs aws command line tools for your user)</p>

<p>The cortical results were made from fMRIPprep 1.1.4, installed in a docker container</p>
<p>The hippocampal results were made from fMRIprep 1.5.6, and run via a singularity image</p>

## Settings files:
<ol>
<li><a href="https://github.com/samsydco/HBN/blob/master/settings.py">settings.py</a>: 
<ul>
<li>sets various path</li>
<li>yesnofun: used in 2_RA_rating.py and 2_rating_comp.py</li>
</ul>
</li>
<li><a href="https://github.com/samsydco/HBN/blob/master/ISC_settings.py">ISC_settings.py</a>: Creates functions used later in ISC code. 
<ul>
<li>make_phenol: creates demographic breakdown for all subjects (Age, Sex, phenotypic PC variables - not really used)</li>
<li>even_out: creates two groups with equal sizes and equal numbers of a demographic variable (e.g. sex)</li>
<li>bottom part of code creates 5 equally sized age groups of subjects</li>
</ul>
</li>
<li><a href="https://github.com/samsydco/HBN/blob/master/HMM_settings.py">HMM_settings.py</a>: sets various variables used for HMM code:
<ul>
  <li>pandas dataframe with un-usable parcels is here (due to low log-likelihood in both age groups).</li>
  <li>FDR_p funciton is here</li>
</ul>
</li>
</ol>

## Order of code is as follows:
<ol>
<li><a href="https://github.com/samsydco/HBN/blob/master/1_aws.py">1_aws.py</a>: 
<ul>
  <li>Download available HBN datasets from aws (from either Rutgers (RU) or CBCI (CitiGroup Cornell Brain Imaging Center).</li>
  <li>Removed subjects who had files missing, and noted what was missing in csv file.</li>
  <li>Edited jsons for fieldmap scans for fmriprep compatibility</li>
  <li>Edited fieldmap scans names for fmriprep compatibility </li>
</ul>
</li>
<li><a href="https://github.com/samsydco/HBN/blob/master/2_RA_rating.py.py">2_RA_rating.py</a> and <a href="https://github.com/samsydco/HBN/blob/master/2_rating_comp.py">2_rating_comp.py</a>:
<ul>
  <li><a href="https://github.com/samsydco/HBN/blob/master/2_RA_rating.py.py">2_RA_rating.py</a>: Allowed RA's to fill in ratings for T1 Scans</li>
  <li><a href="https://github.com/samsydco/HBN/blob/master/2_rating_comp.py">2_rating_comp.py</a>: Compare ratings between RAs: Compile RA ratings into one csv, Manually arbitrate between instances in which one RA said "yes" and another "no." </li>
</ul>
</li>
<li><a href="https://github.com/samsydco/HBN/blob/master/3_fmriprep.py">3_fmriprep.py</a> and <a href="https://github.com/samsydco/HBN/blob/master/3_run_singularity.py">3_run_singularity.py</a>: 
<ul>
  <li><a href="https://github.com/samsydco/HBN/blob/master/3_fmriprep.py">3_fmriprep.py</a>: Run fMRIPprep 1.1.4, installed in a docker container</li>
  <li><a href="https://github.com/samsydco/HBN/blob/master/3_run_singularity.py">3_run_singularity.py</a>: Run fMRIprep 1.5.6, via a singularity image</li>
</ul>
</li>
<li><a href="https://github.com/samsydco/HBN/blob/master/4_Preprocess.py">4_Preprocess.py</a> and <a href="https://github.com/samsydco/HBN/blob/master/4_Preprocess_HPC.py">4_Preprocess_HPC.py</a>:
<ul>
  <li><a href="https://github.com/samsydco/HBN/blob/master/4_Preprocess.py">4_Preprocess.py</a>: regresses out confounds found from fmriprep 1.1.4, and saves data in h5 file for cortex (one per subject).</li>
  <li><a href="https://github.com/samsydco/HBN/blob/master/4_Preprocess_HPC.py">4_Preprocess_HPC.py</a>: regresses out confounds found from fmriprep 1.5.6, labels anterior and posterior hippocampus, and saves data in h5 file for Hippocampus (one per subject).</li>
</ul>
</li>
<li> <a href="https://github.com/samsydco/HBN/blob/master/5_parcellation.py">5_parcellation.py</a>: This creates h5 files for each parcel with the vertices contributing to that parcel and a matrix of size (subjects x vertices x time).
</li>
<li> Use vertices from <a href="https://github.com/samsydco/HBN/blob/master/5_parcellation.py">5_parcellation.py</a> to calculate ISCs and HMMs in each age group: <a href="https://github.com/samsydco/HBN/blob/master/6_ISC.py">6_ISC.py</a> and <a href="https://github.com/samsydco/HBN/blob/master/6_HMM_n_k.py">6_HMM_n_k.py</a>:
<ul>
  <li><a href="https://github.com/samsydco/HBN/blob/master/6_ISC.py">6_ISC.py</a>: Determine difference in ISC and between-group ISC for Youngest and Oldest subjects.
    Do 100 subject-age permutations. (Repeatedly run 1000 more shuffles in parcels where p<0.05 or until p>0)</li>
  <li><a href="https://github.com/samsydco/HBN/blob/master/6_HMM_n_k.py">6_HMM_n_k.py</a>: Determine if there is a significant difference in the number of HMM-derived events between Youngest and Oldest groups.
    Do 100 subject-age permutaitons. (Repeatedly run 1000 more shuffles in parcels where p<0.05 or until p>0)</li>
</ul>
</li>
<li> <a href="https://github.com/samsydco/HBN/blob/master/7_HMM_ll.py">7_HMM_ll.py</a>: 
<ul>
  <li>Find where HMM has a poor fit in either Youngest or Oldest subjects (HMM fit in <a href="https://github.com/samsydco/HBN/blob/master/6_HMM_n_k.py">6_HMM_n_k.py</a>).</li>
  <li>See if there is a difference in the number of events in the remaining parcels</li>
</ul>
</li>
<li> <a href="https://github.com/samsydco/HBN/blob/master/8_HMM_stats.py">8_HMM_stats.py</a>: In remaining parcels:
<ul>
  <li>Run joint-fit HMM on Youngest and Oldest ages. Test on held-out subjects in all age groups</li>
  <li>Look for significant log-likelihood difference between Youngest and Oldest ages in jointly-fit HMMs.</li>
  <li>Look for significant "AUC" difference (HMM-prediction or event-timing) between Youngest and Oldest ages in jointly-fit HMMs.</li>
  <li>Null distribution is age-permuted subject-averaged timecourse.</li>
  <li>Run 100 permutations initially, then continue running code, adding 1000 permutations each time until p>0.05 or p!=0</li>
</ul>
</li>
<li> <a href="https://github.com/samsydco/HBN/blob/master/9_p_check.py">9_p_check.py</a>: Make an h5 file with p and q values for ISC and HMM tests. To be used for plotting on brain.
</li>
<li> Make Figures:
<ul>
  <li><a href="https://github.com/samsydco/HBN/blob/master/10_HMM_AUC.py">10_HMM_AUC.py</a>: Makes AUC plots (both with data and example plots).</li>
  <li><a href="https://github.com/samsydco/HBN/blob/master/10_HMM_ll.py">10_HMM_ll.py</a>: Makes Log Likelihood plots in significant parcels.</li>
</ul>
</li>
</ol>

## Code Relevant for Hippocampus:
<ol>
<li><a href="https://github.com/samsydco/HBN/blob/master/HPC.py">HPC.py</a>: Gets average hippocampus trace for each subject (in both anterior and posterior hippocampus)</li>
<li><a href="https://github.com/samsydco/HBN/blob/master/event_ratings.py">event_ratings.py</a>: Calculates event timecourse from behavioral raters and compares to hippocampus timecourse. Behavioral annotations found in "video_segmentation" folder.</li>
</ol>

## Other code:
<ul>
<li><a href="https://github.com/samsydco/HBN/blob/master/6_ISC_test.py">6_ISC_test.py</a>: Demonstrates that split-half ISC calculates ISC faster than Pairwise ISC or Leave-one-out ISC</li>
<li><a href="https://github.com/samsydco/HBN/blob/master/1__pheno.py">1__pheno.py</a>: Calculates various phenotypic information for sample such as how many subjects were eliminated at various pre-processing steps.</li>
<li><a href="https://github.com/samsydco/HBN/blob/master/HMM_vs_hand.py">HMM_vs_hand.py</a>: Compares HMM event timecourses to behavioral event timecourses</li>
</ul>


