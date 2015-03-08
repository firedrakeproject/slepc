#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#
#  SLEPc is free software: you can redistribute it and/or modify it under  the
#  terms of version 3 of the GNU Lesser General Public License as published by
#  the Free Software Foundation.
#
#  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
#  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
#  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
#  more details.
#
#  You  should have received a copy of the GNU Lesser General  Public  License
#  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import petscconf, log, package

class Trlan(package.Package):

  def __init__(self,argdb,log):
    self.packagename = 'trlan'
    self.havepackage = 0
    self.packagedir  = ''
    self.packagelibs = []
    self.log         = log
    self.ProcessArgs(argdb)

  def Check(self,conf,vars,cmake,tmpdir):

    if petscconf.SCALAR == 'complex':
      self.log.Exit('ERROR: TRLAN is not available for complex scalars.')

    if petscconf.PRECISION != 'double':
      self.log.Exit('ERROR: TRLAN is supported only in double precision.')

    if petscconf.IND64:
      self.log.Exit('ERROR: cannot use external packages with 64-bit indices.')

    functions = ['trlan77']
    if self.packagelibs:
      libs = [self.packagelibs]
    else:
      libs = [['-ltrlan_mpi']]

    if self.packagedir:
      dirs = [self.packagedir]
    else:
      dirs = self.GenerateGuesses('TRLan')

    self.packagelibs = self.FortranLib(tmpdir,conf,vars,cmake,'TRLAN',dirs,libs,functions)
