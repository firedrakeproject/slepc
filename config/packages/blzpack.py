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

class Blzpack(package.Package):

  def __init__(self,argdb):
    self.packagename = 'blzpack'
    self.havepackage = 0
    self.packagedir  = ''
    self.packagelibs = []
    self.ProcessArgs(argdb)

  def Check(self,conf,vars,cmake,tmpdir):
  
    if petscconf.SCALAR == 'complex':
      log.Exit('ERROR: BLZPACK does not support complex numbers.')
  
    if (petscconf.PRECISION != 'single') & (petscconf.PRECISION != 'double'):
      log.Exit('ERROR: BLZPACK is supported only in single or double precision.')
  
    if petscconf.IND64:
      log.Exit('ERROR: cannot use external packages with 64-bit indices.')
  
    if petscconf.PRECISION == 'single':
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
  
    self.packagelibs = self.FortranLib(tmpdir,conf,vars,cmake,'BLZPACK',dirs,libs,functions)
