#!/usr/bin/env python3

# Make html file with all EPI to T1 images

import glob
from settings import *

subs = glob.glob('%ssub*.html'%(fmripreppath))
subs = [s.replace('.html', '') for s in subs]
subs = [s.replace(fmripreppath, '') for s in subs]

htmlfilename=fmripreppath+'QA_EPI_to_T1.html'

f = open(htmlfilename, "w")
f.write("<?xml version=\"1.0\" encoding=\"utf-8\" ?> \
<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\"> \
<html xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\"> \
<head> \
<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" /> \
<meta name=\"generator\" content=\"Docutils 0.12: http://docutils.sourceforge.net/\" /> \
<title></title> \
<script src=\"https://code.jquery.com/jquery-3.3.1.slim.min.js\" integrity=\"sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo\" crossorigin=\"anonymous\"></script> \
<script src=\"https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js\" integrity=\"sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy\" crossorigin=\"anonymous\"></script> \
<link rel=\"stylesheet\" href=\"https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css\" integrity=\"sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO\" crossorigin=\"anonymous\"> \
<style type=\"text/css\"> \
.sub-report-title {} \
.run-title {} \
.elem-desc {} \
.elem-filename {} \
 \
div.elem-image { \
  width: 100%; \
  page-break-before:always; \
} \
\
.elem-image object.svg-reportlet { \
    width: 100%; \
    padding-bottom: 5px; \
} \
body { \
    padding: 65px 10px 10px; \
} \
</style> \
</head> \
<body>") 
for sub in list(subs):
    f.write("<p>============================================")
    f.write("<p>%s DM BBR"%(sub))
    f.write("<div class=\"elem-image\"> \
                            <object class=\"svg-reportlet\" type=\"image/svg+xml\" \
                            data=\"./"+sub+"/figures/"+sub+"_task-movieDM_bold_bbr.svg\"> \
                            Problem loading figure"+sub+"/figures/"+sub+"_task-movieDM_bold_bbr.svg. \
                            If the link below works, please try reloading the report in your browser.</object> \
                            </div>")
    f.write("<p>%s TP BBR"%(sub))
    f.write("<div class=\"elem-image\"> \
                            <object class=\"svg-reportlet\" type=\"image/svg+xml\" \
                            data=\"./"+sub+"/figures/"+sub+"_task-movieTP_bold_bbr.svg\"> \
                            Problem loading figure"+sub+"/figures/"+sub+"_task-movieTP_bold_bbr.svg. \
                            If the link below works, please try reloading the report in your browser.</object> \
                            </div>")
    f.write("<p>%s DM SDC"%(sub))
    f.write("<div class=\"elem-image\"> \
                            <object class=\"svg-reportlet\" type=\"image/svg+xml\" \
                            data=\"./"+sub+"/figures/"+sub+"_task-movieDM_bold_sdc_epi.svg\"> \
                            Problem loading figure"+sub+"/figures/"+sub+"_task-movieDM_bold_sdc_epi.svg. \
                            If the link below works, please try reloading the report in your browser.</object> \
                            </div>")
    f.write("<p>%s TP SDC"%(sub))
    f.write("<div class=\"elem-image\"> \
                            <object class=\"svg-reportlet\" type=\"image/svg+xml\" \
                            data=\"./"+sub+"/figures/"+sub+"_task-movieTP_bold_sdc_epi.svg\"> \
                            Problem loading figure"+sub+"/figures/"+sub+"_task-movieTP_bold_sdc_epi.svg. \
                            If the link below works, please try reloading the report in your browser.</object> \
                            </div>")
f.close()

