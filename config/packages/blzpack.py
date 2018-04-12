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

class Blzpack(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename    = 'blzpack'
    self.installable    = True
    self.supportsscalar = ['real']
    self.supportssingle = True
    self.ProcessArgs(argdb)

  def Check(self,conf,vars,cmake,petsc):
    if petsc.precision == 'single':
      functions = ['blzdrs']
    else:
      functions = ['blzdrd']

    if self.packagelibs:
      libs = [self.packagelibs]
    else:
      libs = [['-lblzpack']]

    if self.packagedir:
      dirs = [self.packagedir]
    else:
      dirs = self.GenerateGuesses('Blzpack')

    self.FortranLib(conf,vars,cmake,dirs,libs,functions)

