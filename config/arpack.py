#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain
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

import os
import sys

import petscconf
import check

def Check(conf,directory,libs):

  if petscconf.SCALAR == 'real':
    if petscconf.PRECISION == 'single':
      functions = ['psnaupd','psneupd','pssaupd','psseupd']
    else:
      functions = ['pdnaupd','pdneupd','pdsaupd','pdseupd']
  else:
    if petscconf.PRECISION == 'single':
      functions = ['pcnaupd','pcneupd']
    else:
      functions = ['pznaupd','pzneupd']

  if libs:
    libs = [libs]
  else:
    libs = [['-lparpack','-larpack'],['-lparpack_MPI','-larpack'],['-lparpack_MPI-LINUX','-larpack_LINUX'],['-lparpack_MPI-SUN4','-larpack_SUN4']]

  if directory:
    dirs = [directory]
  else:
    dirs = check.GenerateGuesses('Arpack')
    
  return check.FortranLib(conf,'ARPACK',dirs,libs,functions)
