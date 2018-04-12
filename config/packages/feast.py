#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import log, package

class Feast(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename    = 'feast'
    self.installable    = True
    self.supportsscalar = ['complex']
    self.supportssingle = True
    self.ProcessArgs(argdb)

  def Check(self,conf,vars,cmake,petsc):
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

    if self.packagelibs:
      libs = [self.packagelibs]
    else:
      if petsc.mpiuni:
        libs = [['-lfeast']]
      else:
        libs = [['-lpfeast']]

    if self.packagedir:
      dirs = [self.packagedir]
    else:
      dirs = self.GenerateGuesses('Feast')

    self.FortranLib(conf,vars,cmake,dirs,libs,functions)

