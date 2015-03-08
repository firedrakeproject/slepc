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

import os
import petscconf, log, package

class Primme(package.Package):

  def __init__(self,argdb):
    self.packagename = 'primme'
    self.havepackage = 0
    self.packagedir  = ''
    self.packagelibs = []
    self.ProcessArgs(argdb)

  def Check(self,conf,vars,cmake,tmpdir):
  
    log.write('='*80)
    log.Println('Checking PRIMME library...')
  
    if petscconf.PRECISION != 'double':
      log.Exit('ERROR: PRIMME is supported only in double precision.')
  
    if petscconf.IND64:
      log.Exit('ERROR: cannot use external packages with 64-bit indices.')
  
    functions_base = ['primme_set_method','primme_Free','primme_initialize']
    if self.packagedir:
      dirs = [self.packagedir]
    else:
      dirs = self.GenerateGuesses('Primme')
  
    libs = self.packagelibs
    if not libs:
      libs = ['-lprimme']
    if petscconf.SCALAR == 'real':
      functions = functions_base + ['dprimme']
    else:
      functions = functions_base + ['zprimme']
  
    for d in dirs:
      if d:
        if 'rpath' in petscconf.SLFLAG:
          l = [petscconf.SLFLAG + d] + ['-L' + d] + libs
        else:
          l = ['-L' + d] + libs
        f = ['-I' + os.path.join(d,'PRIMMESRC','COMMONSRC')]
      else:
        l =  libs
        f = []
      if self.Link(tmpdir,functions,[],l+f):
        conf.write('#ifndef SLEPC_HAVE_PRIMME\n#define SLEPC_HAVE_PRIMME 1\n#endif\n\n')
        vars.write('PRIMME_LIB = ' + ' '.join(l) + '\n')
        vars.write('PRIMME_FLAGS = ' + ' '.join(f) + '\n')
        cmake.write('set (SLEPC_HAVE_PRIMME YES)\n')
        cmake.write('find_library (PRIMME_LIB primme HINTS '+ d +')\n')
        cmake.write('find_path (PRIMME_INCLUDE primme.h ' + d + '/PRIMMESRC/COMMONSRC)\n')
        self.packagelibs = l+f
        return
  
    log.Println('ERROR: Unable to link with PRIMME library')
    log.Println('ERROR: In directories '+' '.join(dirs))
    log.Println('ERROR: With flags '+' '.join(libs))
    log.Exit('')
