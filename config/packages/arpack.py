#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain
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

import log, package

class Arpack(package.Package):

  def __init__(self,argdb,log):
    self.packagename  = 'arpack'
    self.downloadable = False
    self.packagedir   = ''
    self.packagelibs  = []
    self.log          = log
    self.ProcessArgs(argdb)

  def Check(self,conf,vars,cmake,petsc):

    if (petsc.precision != 'single') & (petsc.precision != 'double'):
      self.log.Exit('ERROR: ARPACK is supported only in single or double precision.')

    if petsc.ind64:
      self.log.Exit('ERROR: Cannot use external packages with 64-bit indices.')

    if petsc.mpiuni:
      if petsc.scalar == 'real':
        if petsc.precision == 'single':
          functions = ['snaupd','sneupd','ssaupd','sseupd']
        else:
          functions = ['dnaupd','dneupd','dsaupd','dseupd']
      else:
        if petsc.precision == 'single':
          functions = ['cnaupd','cneupd']
        else:
          functions = ['znaupd','zneupd']
    else:
      if petsc.scalar == 'real':
        if petsc.precision == 'single':
          functions = ['psnaupd','psneupd','pssaupd','psseupd']
        else:
          functions = ['pdnaupd','pdneupd','pdsaupd','pdseupd']
      else:
        if petsc.precision == 'single':
          functions = ['pcnaupd','pcneupd']
        else:
          functions = ['pznaupd','pzneupd']

    if self.packagelibs:
      libs = [self.packagelibs]
    else:
      if petsc.mpiuni:
        libs = [['-larpack'],['-larpack_LINUX'],['-larpack_SUN4']]
      else:
        libs = [['-lparpack','-larpack'],['-lparpack_MPI','-larpack'],['-lparpack_MPI-LINUX','-larpack_LINUX'],['-lparpack_MPI-SUN4','-larpack_SUN4']]

    if self.packagedir:
      dirs = [self.packagedir]
    else:
      dirs = self.GenerateGuesses('Arpack')

    self.FortranLib(conf,vars,cmake,dirs,libs,functions)

