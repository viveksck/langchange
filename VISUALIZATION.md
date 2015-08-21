# Instructions for visualization semantic change

### Author: William Hamilton (wleif@stanford.edu)

## Overview

This document assumes that the core code base (described in the README file) is compiled and working.
The instructions here are specific to the low-dimensional visualization code.

## Dependencies

The three major dependencies for these packages are:

  * bokeh: http://bokeh.pydata.org/en/latest/docs/installation.html
  * sklearn: http://scikit-learn.org/stable/
  * cython: http://docs.cython.org/src/quickstart/install.html

The last two should be installed already but you may need to download bokeh.

## Visualization tools

There are two visualization tools for analyzing semantic change, both of which rely on the bokeh package:

  * `path_vis.py`: allows one to visualize the path that a word moves through semantic space starting at the year 1900.
  * `neigh_vis.py`: allows one to visualize how a word's neighborhood changes over time. 

Both of these tools allow the user to specify the target word, what year to visualize, and what year to use to construct the linear visualization basis.
To run these tools you first must download historical vectors from `http://web.stanford.edu/~wleif/data/histvecs.tar.gz` and you must alter the `INPUT_PATH` variable in the `sequentialembedding.py` file to point to the path to these vectors. Following this, you can use the command:

    bokeh-server --script [script-name]

where `[script-name]` is one of `pathvis.py` or `neighvis.py`.
This will open a server on port 5006 of your localhost IP address. 
`pathvis.py` will be accessible as `localhost:5006/bokeh/pathchange` while `neigh_vis.py` will be accessible at `localhost:5006/bokeh/neighchange`.
For more information on bokeh options run `bokeh-server --help`.
