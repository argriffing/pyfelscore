pyfelscore
==========

This is like a bad version of
[beagle-lib](http://code.google.com/p/beagle-lib/).

The point of this module is to provide a faster-than-pure-python
implementation of Felsenstein's dynamic programming algorithm
for computing a log likelihood given
 * a rooted combinatorial tree structure relating some taxa
 * a matrix of state transition probabilities for each edge of the tree
 * a prior finite distribution over states at the root of the tree
 * i.i.d vectors of joint state observations on some subset of the
   taxa (e.g. at the leaves of the tree).

This is a Cython module which is meant to be buried in a more useful library.
The API is based on NumPy ndarrays,
so this module does not know anything about parsing
tree or alignment files (e.g. Newick, Nexus, Phylip, NeXML).


Requirements
------------

 * [Python](http://python.org/) 2.7+ (but it is not tested on 3.x)
 * [NumPy](http://www.numpy.org/)
 * a computing environment that knows how to compile C code


Installation
------------

The easiest way to install this module is to use
[pip](http://www.pip-installer.org/)
to install directly from github.

`pip install --user https://github.com/argriffing/pyfelscore/zipball/master`

This does not not require administrator privileges,
and it can be easily reverted using the command

`pip uninstall pyfelscore`

The installation should be very easy with Linux-based operating systems,
more difficult with OS X,
and probably not worth attempting on Windows.

