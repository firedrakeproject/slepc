#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import log, package

class Feast(package.Package):

  def __init__(self,argdb,log,petscpackages):
    package.Package.__init__(self,argdb,log)
    self.packagename    = 'feast'
    self.installable    = True
    self.petscdepend    = 'mkl'
    self.supportssingle = True
    self.supports64bint = True
    self.ProcessArgs(argdb,petscpackages)

  def Check(self,slepcconf,slepcvars,petsc,archdir):
    if not 'mkl' in petsc.packages:
      self.log.Exit('The FEAST interface requires that PETSc has been built with Intel MKL')
    functions = ['feastinit']
    if petsc.scalar == 'real':
      if petsc.precision == 'single':
        functions += ['sfeast_srci']
      else:
        functions += ['dfeast_srci']
    else:
      if petsc.precision == 'single':
        functions += ['cfeast_hrci']
      else:
        functions += ['zfeast_hrci']

    (result,output) = self.Link(functions,[],[])
    if not result:
      self.log.write('WARNING: Unable to link with FEAST, maybe your MKL version does not contain it')
      self.log.write('If you do not want to check for FEAST, rerun configure adding --with-feast=0')
      self.havepackage = False
    else:
      slepcconf.write('#define SLEPC_HAVE_FEAST 1\n')
      self.havepackage = True

