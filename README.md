# The langchange package: Systematically analyzing semantic change 

### Author: William Hamilton (wleif@stanford.edu)

## Overview

This package contains a collection of scripts for analyzing semantic change using distributional statistics.
Two visualization tools are accessible in the main directory, and instructions for running them are below. 
The sub-directories are structured as follows:
  * googlengrams: scripts for pulling, processing, and storing data from the Google N-Gram dataset.
  * statutils: general purpose statistical utilities for analyzing distributional statistics.
  * modglove: a modified version of Geoffrey Pennington's GloVe code. See the local README for citations and more information.
  * vecanalysis: utilities for representing and analyzing distributional word vector embeddings.
  * misc: miscellaneous helper scripts
  * cluster: scripts for running on Torque scheduled cluster.

## Dependencies

The three major dependencies for these packages are:

  * bokeh: http://bokeh.pydata.org/en/latest/docs/installation.html
  * sklearn: http://scikit-learn.org/stable/
  * cython: http://docs.cython.org/src/quickstart/install.html

Installing these three packages should give you everything you need.

## Visualization tools

There are two visualization tools for analyzing semantic change, both of which rely on the bokeh package:

  * `path_vis.py`: allows one to visualize the path that a word moves through semantic space starting at the year 1900.
  * `neigh_vis.py`: allows one to visualize how a word's neighborhood changes over time. 

Both of these tools allow the user to specify the target word, what year to visualize, and what year to use to construct the linear visualization basis.
To run these tools, use the command:

    bokeh-server --script [script-name]

This will open a server on port 5006 of your localhost IP address. 
`pathvis.py` will be accessible as `localhost:5006/bokeh/pathchange` while `neigh_vis.py` will be accessible at `localhost:5006/bokeh/neighchange`.
For more information on bokeh options run `bokeh-server --help`.

