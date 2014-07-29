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

import sys

import petscconf

def Open(filename):
  global f
  f = open(filename,'w')
  return

def Println(string):
  print string
  f.write(string)
  f.write('\n')

def Print(string):
  print string,
  f.write(string+' ')

def write(string):
  f.write(string)
  f.write('\n')

def Exit(string):
  f.write(string)
  f.write('\n')
  f.close()
  print string
  sys.exit('ERROR: See "' + petscconf.ARCH + '/conf/configure.log" file for details')

