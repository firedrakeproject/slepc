SLEPc for Python
================

Python bindings for SLEPc.

Install
-------

If you have a working MPI implementation and the ``mpicc`` compiler
wrapper is on your search path, it highly recommended to install
``mpi4py`` first::

  $ pip install mpi4py

Ensure you have NumPy installed::

  $ pip install numpy

and finally::

  $ pip install petsc petsc4py slepc slepc4py

You can also install the in-development version of slepc4py with::

  $ pip install Cython numpy mpi4py
  $ pip install --no-deps git+https://gitlab.com/petsc/petsc
  $ pip install --no-deps git+https://bitbucket.org/petsc/petsc4py
  $ pip install --no-deps git+https://gitlab.com/slepc/slepc
  $ pip install --no-deps git+https://gitlab.com/slepc/slepc4py

or::

  $ pip install Cython numpy mpi4py
  $ pip install --no-deps https://gitlab.com/petsc/petsc/-/archive/master.tar.gz
  $ pip install --no-deps https://bitbucket.org/petsc/petsc4py/get/master.tar.gz
  $ pip install --no-deps https://gitlab.com/slepc/slepc/-/archive/master.tar.gz
  $ pip install --no-deps https://gitlab.com/slepc/slepc4py/-/archive/master.tar.gz


Citations
---------

If SLEPc for Python been significant to a project that leads to an
academic publication, please acknowledge that fact by citing the
project.

* L. Dalcin, P. Kler, R. Paz, and A. Cosimo,
  *Parallel Distributed Computing using Python*,
  Advances in Water Resources, 34(9):1124-1139, 2011.
  https://doi.org/10.1016/j.advwatres.2011.04.013

* V. Hernandez, J.E. Roman and V. Vidal.
  *SLEPc: A Scalable and Flexible Toolkit for the
  Solution of Eigenvalue Problems*,
  ACM Transactions on Mathematical Software, 31(3):351-362, 2005.
  https://doi.org/10.1145/1089014.1089019
