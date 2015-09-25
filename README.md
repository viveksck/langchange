#Statistically Analyzing Semantic Change

### Author: William Hamilton (wleif@stanford.edu)

## Overview 

This code base contains an eclectic collection of tool for analyzing semantic change using vector space semantics.
The structure of the code (in terms of folder organization) is as follows:

* `cooccurrence` contains core tools for analyzing word cooccurrences, i.e. generating word vectors and running batch statistics on word vectors.
* `googlengram` contains code for pulling and processing the Google N-Gram Data.
* `statutils` contains helper code for common statistical tasks.
* `vecanalysis` contains code that provides a high-level interface to (historical) word vectors and is originally based upon Omar Levy's hyperwords package (https://bitbucket.org/omerlevy/hyperwords).
* `modglove` contains a modified version of Jeffrey Pennington's GloVe code (http://nlp.stanford.edu/projects/glove/)
* `cluster` contains scripts for running code on PBS cluster.
* `statsmodels` is a snapshot of a development version of statsmodels (https://github.com/statsmodels/statsmodels/), which I use in my code. The snapshot is included to ease replication of my results but note that I have not made alterations to this code nor do I claim any ownership of it.
* `notebooks` contains notebooks useful for replicating my published results

See REPLICATION.md for detailed instructions on how to replicate specific published/submitted results.

See VISUALIZATION.md for detailed instructions on using my visualization tools, which allow you to visualize semantic change at the individual word level using low-dimensional word vector embeddings.

## Dependencies

Core dependencies:
  * sklearn: http://scikit-learn.org/stable/
  * cython: http://docs.cython.org/src/quickstart/install.html

As noted above, a local version of statsmodels is included for replication purposes, so there is no need to manually install this dependency.

For visualization and running the notebooks:
  * seaborn: http://stanford.edu/~mwaskom/software/seaborn/

## Installation

After the core dependencies are installed run

    python setup.py build_ext --inplace

to compile the necessary cython modules.

