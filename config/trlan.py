#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain
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

def Check(conf,vars,cmake,tmpdir,directory,libs):

  if petscconf.SCALAR == 'complex':
    sys.exit('ERROR: TRLAN does not support complex numbers.') 

  if petscconf.PRECISION == 'single':
    sys.exit('ERROR: TRLAN does not support single precision.') 

  functions = ['trlan77']
  if libs:
    libs = [libs]
  else:
    libs = [['-ltrlan_mpi']]

  if directory:
    dirs = [directory]
  else:
    dirs = check.GenerateGuesses('TRLan')
    
  return check.FortranLib(tmpdir,conf,vars,cmake,'TRLAN',dirs,libs,functions)
