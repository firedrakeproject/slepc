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
import sys
import commands

import petscconf
import log

def LinkWithOutput(tmpdir,functions,callbacks,flags):
  code = '#include "petscksp.h"\n'
  for f in functions:
    code += 'PETSC_EXTERN int\n' + f + '();\n'

  for c in callbacks:
    code += 'int '+ c + '() { return 0; } \n'

  code += 'int main() {\n'
  code += 'Vec v; Mat m; KSP k;\n'
  code += 'PetscInitializeNoArguments();\n'
  code += 'VecCreate(PETSC_COMM_WORLD,&v);\n'
  code += 'MatCreate(PETSC_COMM_WORLD,&m);\n'
  code += 'KSPCreate(PETSC_COMM_WORLD,&k);\n'
  for f in functions:
    code += f + '();\n'
  code += 'return 0;\n}\n'

  cfile = open(os.sep.join([tmpdir,'checklink.c']),'w')
  cfile.write(code)
  cfile.close()
  (result, output) = commands.getstatusoutput('cd ' + tmpdir + ';' + petscconf.MAKE + ' checklink TESTFLAGS="'+str.join(' ',flags)+'"')
  try: os.remove('checklink.c')
  except OSError: pass
  if result:
    return (0,code + output)
  else:
    return (1,code + output)

def Link(tmpdir,functions,callbacks,flags):
  (result, output) = LinkWithOutput(tmpdir,functions,callbacks,flags)
  log.write(output)
  return result

def FortranLink(tmpdir,functions,callbacks,flags):
  output =  '\n=== With linker flags: '+str.join(' ',flags)

  f = []
  for i in functions:
    f.append(i+'_')
  c = []
  for i in callbacks:
    c.append(i+'_')
  (result, output1) = LinkWithOutput(tmpdir,f,c,flags)
  output1 = '\n====== With underscore Fortran names\n' + output1
  if result: return ('UNDERSCORE',output1)

  f = []
  for i in functions:
    f.append(i.upper())
  c = []
  for i in callbacks:
    c.append(i.upper())
  (result, output2) = LinkWithOutput(tmpdir,f,c,flags)
  output2 = '\n====== With capital Fortran names\n' + output2
  if result: return ('CAPS',output2)

  (result, output3) = LinkWithOutput(tmpdir,functions,callbacks,flags)
  output3 = '\n====== With unmodified Fortran names\n' + output3
  if result: return ('STDCALL',output3)

  return ('',output + output1 + output2 + output3)

def GenerateGuesses(name):
  installdirs = ['/usr/local','/opt']
  if 'HOME' in os.environ:
    installdirs.insert(0,os.environ['HOME'])

  dirs = []
  for i in installdirs:
    dirs = dirs + [i + '/lib']
    for d in [name,name.upper(),name.lower()]:
      dirs = dirs + [i + '/' + d]
      dirs = dirs + [i + '/' + d + '/lib']
      dirs = dirs + [i + '/lib/' + d]

  for d in dirs[:]:
    if not os.path.exists(d):
      dirs.remove(d)
  dirs = [''] + dirs
  return dirs

def FortranLib(tmpdir,conf,vars,cmake,name,dirs,libs,functions,callbacks = []):
  log.write('='*80)
  log.Println('Checking '+name+' library...')

  error = ''
  mangling = ''
  for d in dirs:
    for l in libs:
      if d:
        if 'rpath' in petscconf.SLFLAG:
          flags = [petscconf.SLFLAG + d] + ['-L' + d] + l
        else:
          flags = ['-L' + d] + l
      else:
        flags = l
      (mangling, output) = FortranLink(tmpdir,functions,callbacks,flags)
      error += output
      if mangling: break
    if mangling: break

  if mangling:
    log.write(output)
  else:
    log.write(error)
    log.Println('ERROR: Unable to link with library '+ name)
    log.Println('ERROR: In directories '+''.join([s+' ' for s in dirs]))
    log.Println('ERROR: With flags '+''.join([s+' ' for s in flags]))
    log.Exit('')

  conf.write('#ifndef SLEPC_HAVE_' + name + '\n#define SLEPC_HAVE_' + name + ' 1\n#define SLEPC_' + name + '_HAVE_'+mangling+' 1\n#endif\n\n')
  vars.write(name + '_LIB = '+str.join(' ',flags)+'\n')
  cmake.write('set (SLEPC_HAVE_' + name + ' YES)\n')
  libname = ''.join([s.lstrip('-l')+' ' for s in l])
  cmake.write('set (' + name + '_LIB "")\nforeach (libname ' + libname + ')\n  string (TOUPPER ${libname} LIBNAME)\n  find_library (${LIBNAME}LIB ${libname} HINTS '+ d +')\n  list (APPEND ' + name + '_LIB "${${LIBNAME}LIB}")\nendforeach()\n')
  return flags
