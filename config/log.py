#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     SLEPc - Scalable Library for Eigenvalue Problem Computations
#     Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain
#
#     This file is part of SLEPc. See the README file for conditions of use
#     and additional information.
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
  
def Write(string):
  f.write(string)
  f.write('\n')
  
def Exit(string):
  f.write(string)
  f.write('\n')
  f.close()
  print string
  sys.exit('ERROR: See "configure_log_' + petscconf.ARCH + '" file for details')

