## Instructions for Replicating "Statistical regularities in the evolution of the English language."

The easiest way to replicate the results is as follows:

1) Download statistics files from
  `http://stanford.edu/~wleif/files/stats_lang_data.tar.gz` -- contains word vectors and summary statistics.
  `http://stanford.edu/~wleif/files/misc_lang_data.tar.gz` --- contains POS tags and other misc. info

Place these files in some known directories, `DATA_DIR` and `MISC_DIR` respectively. 
 
2) All the plots etc. from the paper can be replicated using the IPython notebooks under the `notebooks` dir. You will need to set the correct directory names in the first cell. At this time, the documentation is somewhat sparse but all the code used in the paper is available. 
* `reganalysis.ipynb` contains code for learning mixed models and analysing the effect of frequency and breadth. Running all the cells will replicate Figures 1 and 2, Extended Data Figures 1, 2, and 3, and Extended Data Table 1.
* `newwords.ipynb` contains code for analyzing the introduction of new words. Running all the cells will replicate Figure 3.
* `prediction.ipynb` contains code for predicting how much word meanings will change. Running all the cells will replicate Figure 4 and the two parts of Extended Data Figure 4.

3) If you wish to replicate the whole analysis pipeline. You will need to 
    
    * Use the scripts under `googlengrams/pullscripts` to pull data from GoogleBooks (make sure you have >30Gbs of free space!!). Or else pull the data yourself. The data needs to eventually be in a binary "dok" sparse matrix format. See `cooccurrence/matstore.pyx` for details.
    * Use the scripts under `cooccurrence` to build vector representations. In particular, `runsymconf.py` to get binary confidence matrix from co-occurrence counts and then `runlaplaceppmigen.py` to get PPMI word vectors. See these files for details. 
    * Run the stats scripts under `cooccurrence` to get statistics from the vectors, e.g. `runnetstats.py` to get most statistics. 

4) Email wleif@stanford.edu if something is off or missing!
