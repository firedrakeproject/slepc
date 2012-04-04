#!/usr/bin/env python
#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain
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
import time
import commands
import tempfile
import shutil

# Use en_US as language so that compiler messages are in English
if 'LC_LOCAL' in os.environ and os.environ['LC_LOCAL'] != '' and os.environ['LC_LOCAL'] != 'en_US' and os.environ['LC_LOCAL']!= 'en_US.UTF-8': os.environ['LC_LOCAL'] = 'en_US.UTF-8'
if 'LANG' in os.environ and os.environ['LANG'] != '' and os.environ['LANG'] != 'en_US' and os.environ['LANG'] != 'en_US.UTF-8': os.environ['LANG'] = 'en_US.UTF-8'

# should be run from the toplevel
configDir = os.path.abspath('config')
if not os.path.isdir(configDir):
  raise RuntimeError('Run configure from $SLEPC_DIR, not '+os.path.abspath('.'))
sys.path.insert(0, configDir)

import petscversion
import petscconf
import log
import check
import arpack
import blzpack
import trlan  
import lapack
import primme
import blopex
import slepc4py

if not hasattr(sys, 'version_info') or not sys.version_info[0] == 2 or not sys.version_info[1] >= 4:
  print '*****  You must have Python2 version 2.4 or higher to run ./configure.py   ******'
  print '*           Python is easy to install for end users or sys-admin.               *'
  print '*                   http://www.python.org/download/                             *'
  print '*                                                                               *'
  print '*            You CANNOT configure SLEPc without Python                          *'
  print '*********************************************************************************'
  sys.exit(4)

# support a few standard configure option types 
for l in range(1,len(sys.argv)):
  name = sys.argv[l]
  if name.startswith('--enable'):
    sys.argv[l] = name.replace('--enable','--with')
    if name.find('=') == -1: sys.argv[l] += '=1'
  if name.startswith('--disable'):
    sys.argv[l] = name.replace('--disable','--with')
    if name.find('=') == -1: sys.argv[l] += '=0'
    elif name.endswith('=1'): sys.argv[l].replace('=1','=0')
  if name.startswith('--without'):
    sys.argv[l] = name.replace('--without','--with')
    if name.find('=') == -1: sys.argv[l] += '=0'
    elif name.endswith('=1'): sys.argv[l].replace('=1','=0')

# Check configure parameters
havearpack = 0
arpackdir = ''
arpacklibs = []
haveblzpack = 0
blzpackdir = ''
blzpacklibs = []
havetrlan = 0
trlandir = ''
trlanlibs = []
haveprimme = 0
primmedir = ''
primmelibs = []
getblopex = 0
haveblopex = 0
blopexurl = ''
getslepc4py = 0
prefixdir = ''

for i in sys.argv[1:]:
  if   i.startswith('--with-arpack-dir='):
    arpackdir = i.split('=')[1]
    havearpack = 1
  elif i.startswith('--with-arpack-flags='):
    arpacklibs = i.split('=')[1].split(',')
    havearpack = 1
  elif i.startswith('--with-arpack'):
    havearpack = not i.endswith('=0')
  elif i.startswith('--with-blzpack-dir='):
    blzpackdir = i.split('=')[1]
    haveblzpack = 1
  elif i.startswith('--with-blzpack-flags='):
    blzpacklibs = i.split('=')[1].split(',')
    haveblzpack = 1
  elif i.startswith('--with-blzpack'):
    haveblzpack = not i.endswith('=0')
  elif i.startswith('--with-trlan-dir='):
    trlandir = i.split('=')[1]
    havetrlan = 1
  elif i.startswith('--with-trlan-flags='):
    trlanlibs = i.split('=')[1].split(',')
    havetrlan = 1
  elif i.startswith('--with-trlan'):
    havetrlan = not i.endswith('=0')
  elif i.startswith('--with-primme-dir'):
    primmedir = i.split('=')[1]
    haveprimme = 1
  elif i.startswith('--with-primme-flags='):
    primmelibs = i.split('=')[1].split(',')
    haveprimme = 1
  elif i.startswith('--with-primme'):
    haveprimme = not i.endswith('=0')
  elif i.startswith('--download-blopex'):
    getblopex = not i.endswith('=0')
    try: blopexurl = i.split('=')[1]
    except IndexError: pass
  elif i.startswith('--download-slepc4py'):
    getslepc4py = not i.endswith('=0')
  elif i.startswith('--prefix='):
    prefixdir = i.split('=')[1]
  elif i.startswith('--h') or i.startswith('-h') or i.startswith('-?'):
    print 'SLEPc Configure Help'
    print '-'*80
    print '  --prefix=<dir>                   : Specify location to install SLEPc (e.g., /usr/local)'
    print 'ARPACK:'
    print '  --with-arpack                    : Indicate if you wish to test for ARPACK (PARPACK)'
    print '  --with-arpack-dir=<dir>          : Indicate the directory for ARPACK libraries'
    print '  --with-arpack-flags=<flags>      : Indicate comma-separated flags for linking ARPACK'
    print 'BLZPACK:'
    print '  --with-blzpack                   : Indicate if you wish to test for BLZPACK'
    print '  --with-blzpack-dir=<dir>         : Indicate the directory for BLZPACK libraries'
    print '  --with-blzpack-flags=<flags>     : Indicate comma-separated flags for linking BLZPACK'
    print 'TRLAN:'
    print '  --with-trlan                     : Indicate if you wish to test for TRLAN'
    print '  --with-trlan-dir=<dir>           : Indicate the directory for TRLAN libraries'
    print '  --with-trlan-flags=<flags>       : Indicate comma-separated flags for linking TRLAN'
    print 'PRIMME:'
    print '  --with-primme                    : Indicate if you wish to test for PRIMME'
    print '  --with-primme-dir=<dir>          : Indicate the directory for PRIMME libraries'
    print '  --with-primme-flags=<flags>      : Indicate comma-separated flags for linking PRIMME'
    print 'BLOPEX:'
    print '  --download-blopex                : Download and install BLOPEX in SLEPc directory'
    print 'slepc4py:'
    print '  --download-slepc4py              : Download and install slepc4py in SLEPc directory'
    sys.exit(0)
  else:
    sys.exit('ERROR: Invalid argument ' + i +'. Use -h for help')

prefixinstall = not prefixdir==''

# Check if enviroment is ok
print 'Checking environment...'
if 'SLEPC_DIR' in os.environ:
  slepcdir = os.environ['SLEPC_DIR']
  if not os.path.exists(slepcdir) or not os.path.exists(os.sep.join([slepcdir,'config'])):
    sys.exit('ERROR: SLEPC_DIR enviroment variable is not valid')
  if os.path.realpath(os.getcwd()) != os.path.realpath(slepcdir):
    sys.exit('ERROR: SLEPC_DIR is not the current directory')
else:
  slepcdir = os.getcwd();
  if not os.path.exists(os.sep.join([slepcdir,'config'])):
    sys.exit('ERROR: Current directory is not valid')

if 'PETSC_DIR' in os.environ:
  petscdir = os.environ['PETSC_DIR']
  if not os.path.exists(petscdir):
    sys.exit('ERROR: PETSC_DIR enviroment variable is not valid')
else:
  if prefixdir:
    petscdir = prefixdir
    os.environ['PETSC_DIR'] = petscdir
  else:
    sys.exit('ERROR: PETSC_DIR enviroment variable is not set')

# Check PETSc version
petscversion.Load(petscdir)
if petscversion.VERSION < '3.2':
  sys.exit('ERROR: This SLEPc version is not compatible with PETSc version '+petscversion.VERSION) 

# Check some information about PETSc configuration
petscconf.Load(petscdir)
if not petscconf.PRECISION in ['double','single','__float128']:
  sys.exit('ERROR: This SLEPc version does not work with '+petscconf.PRECISION+' precision')
if prefixinstall and not petscconf.ISINSTALL:
  sys.exit('ERROR: SLEPc cannot be configured for non-source installation if PETSc is not configured in the same way.')

# Check whether this is a working copy of the Subversion repository
subversion = 0
if os.path.exists(os.sep.join([slepcdir,'src','docs'])) and os.path.exists(os.sep.join([slepcdir,'.svn'])):
  (result, output) = commands.getstatusoutput('svn info')
  if result:
    print 'WARNING: SLEPC_DIR appears to be a subversion working copy, but svn is not found in PATH'
  else:
    subversion = 1
    svnrev = '-1'
    svndate = '-1'
    for line in output.split('\n'):
      if line.startswith('Last Changed Rev: '):
        svnrev = line.split('Rev: ')[-1]
      if line.startswith('Last Changed Date: '):
        svndate = line.split('Date: ')[-1]

# Create architecture directory and configuration files
archdir = os.sep.join([slepcdir,petscconf.ARCH])
if not os.path.exists(archdir):
  try:
    os.mkdir(archdir)
  except:
    sys.exit('ERROR: cannot create architecture directory ' + archdir)
confdir = os.sep.join([archdir,'conf'])
if not os.path.exists(confdir):
  try:
    os.mkdir(confdir)
  except:
    sys.exit('ERROR: cannot create configuration directory ' + confdir)
incdir = os.sep.join([archdir,'include'])
if not os.path.exists(incdir):
  try:
    os.mkdir(incdir)
  except:
    sys.exit('ERROR: cannot create include directory ' + incdir)
libdir = os.sep.join([archdir,'lib'])
if not os.path.exists(libdir):
  try:
    os.mkdir(libdir)
  except:
    sys.exit('ERROR: cannot create lib directory ' + libdir)
try:
  slepcvars = open(os.sep.join([confdir,'slepcvariables']),'w')
  if not prefixdir:
    prefixdir = archdir
  slepcvars.write('SLEPC_DESTDIR = ' + prefixdir +'\n')
  testruns = set(petscconf.TEST_RUNS.split())
  testruns = testruns.intersection(set(['C','F90','Fortran','C_NoComplex','Fortran_NoComplex']))
  slepcvars.write('TEST_RUNS = ' + ' '.join(testruns) +'\n')
except:
  sys.exit('ERROR: cannot create configuration file in ' + confdir)
try:
  slepcrules = open(os.sep.join([confdir,'slepcrules']),'w')
except:
  sys.exit('ERROR: cannot create rules file in ' + confdir)
try:
  slepcconf = open(os.sep.join([incdir,'slepcconf.h']),'w')
  slepcconf.write('#if !defined(__SLEPCCONF_H)\n')
  slepcconf.write('#define __SLEPCCONF_H\n\n')
  if subversion:
    slepcconf.write('#ifndef SLEPC_VERSION_SVN\n#define SLEPC_VERSION_SVN ' + svnrev + '\n#endif\n\n')
    slepcconf.write('#ifndef SLEPC_VERSION_DATE_SVN\n#define SLEPC_VERSION_DATE_SVN "' + svndate + '"\n#endif\n\n')
  slepcconf.write('#ifndef SLEPC_LIB_DIR\n#define SLEPC_LIB_DIR "' + prefixdir + '/lib"\n#endif\n\n')
except:
  sys.exit('ERROR: cannot create configuration header in ' + confdir)
try:
  cmake = open(os.sep.join([confdir,'SLEPcConfig.cmake']),'w')
except:
  sys.exit('ERROR: cannot create CMake configuration file in ' + confdir)
if prefixinstall and os.path.isfile(os.sep.join([prefixdir,'include','slepc.h'])):
  sys.exit('ERROR: prefix directory ' + prefixdir + ' contains files from a previous installation')

# Create temporary directory and makefile for running tests
try:
  tmpdir = tempfile.mkdtemp(prefix='slepc-')
  if not os.path.isdir(tmpdir): os.mkdir(tmpdir)
except:
  sys.exit('ERROR: cannot create temporary directory')
try:
  makefile = open(os.sep.join([tmpdir,'makefile']),'w')
  makefile.write('checklink: checklink.o chkopts\n')
  makefile.write('\t${CLINKER} -o checklink checklink.o ${TESTFLAGS} ${PETSC_KSP_LIB}\n')
  makefile.write('\t@${RM} -f checklink checklink.o\n')
  makefile.write('LOCDIR = ./\n')
  makefile.write('include ${PETSC_DIR}/conf/variables\n')
  makefile.write('include ${PETSC_DIR}/conf/rules\n')
  makefile.close()
except:
  sys.exit('ERROR: cannot create makefile in temporary directory')

# Open log file
log.Open(os.sep.join([confdir,'configure.log']))
log.write('='*80)
log.write('Starting Configure Run at '+time.ctime(time.time()))
log.write('Configure Options: '+str.join(' ',sys.argv))
log.write('Working directory: '+os.getcwd())
log.write('Python version:\n' + sys.version)
log.write('make: ' + petscconf.MAKE)
log.write('PETSc source directory: ' + petscdir)
log.write('PETSc install directory: ' + petscconf.DESTDIR)
log.write('PETSc version: ' + petscversion.VERSION)
log.write('PETSc architecture: ' + petscconf.ARCH)
log.write('SLEPc source directory: ' + slepcdir)
log.write('SLEPc install directory: ' + prefixdir)
log.write('='*80)

# Check if PETSc is working
log.Println('Checking PETSc installation...')
if petscversion.VERSION > '3.2':
  log.Println('WARNING: PETSc version '+petscversion.VERSION+' is newer than SLEPc version')
if petscversion.RELEASE != '1':
  log.Println('WARNING: using PETSc development version')
if petscconf.ISINSTALL:
  if os.path.realpath(petscconf.DESTDIR) != os.path.realpath(petscdir):
    log.Println('WARNING: PETSC_DIR does not point to PETSc installation path')
if not check.Link(tmpdir,[],[],[]):
  log.Exit('ERROR: Unable to link with PETSc')

# Check for external packages
if havearpack:
  arpacklibs = arpack.Check(slepcconf,slepcvars,cmake,tmpdir,arpackdir,arpacklibs)
if haveblzpack:
  blzpacklibs = blzpack.Check(slepcconf,slepcvars,cmake,tmpdir,blzpackdir,blzpacklibs)
if havetrlan:
  trlanlibs = trlan.Check(slepcconf,slepcvars,cmake,tmpdir,trlandir,trlanlibs)
if haveprimme:
  primmelibs = primme.Check(slepcconf,slepcvars,cmake,tmpdir,primmedir,primmelibs)
if getblopex:
  blopexlibs = blopex.Install(slepcconf,slepcvars,cmake,tmpdir,blopexurl,archdir)
  haveblopex = 1

# Check for missing LAPACK functions
missing = lapack.Check(slepcconf,slepcvars,cmake,tmpdir)

# Download and install slepc4py
if getslepc4py:
  slepc4py.Install()
slepc4py.addMakeRule(slepcrules,prefixdir,prefixinstall,getslepc4py)

# Make Fortran stubs if necessary
if subversion and hasattr(petscconf,'FC'):
  try:
    import generatefortranstubs
    generatefortranstubs.main(petscconf.BFORT)
  except AttributeError:
    sys.exit('ERROR: cannot generate Fortran stubs; try configuring PETSc with --download-sowing or use a mercurial version of PETSc')

# CMake stuff
cmake.write('set (SLEPC_PACKAGE_LIBS "${ARPACK_LIB}" "${BLZPACK_LIB}" "${TRLAN_LIB}" "${PRIMME_LIB}" "${BLOPEX_LIB}" )\n')
cmake.write('set (SLEPC_PACKAGE_INCLUDES "${PRIMME_INCLUDE}")\n')
cmake.write('find_library (PETSC_LIB petsc HINTS ${PETSc_BINARY_DIR}/lib )\n')
cmake.write('''
if (NOT PETSC_LIB) # Interpret missing libpetsc to mean that PETSc was built --with-single-library=0
  set (PETSC_LIB "")
  foreach (pkg sys vec mat dm ksp snes ts)
    string (TOUPPER ${pkg} PKG)
    find_library(PETSC${PKG}_LIB "petsc${pkg}" HINTS ${PETSc_BINARY_DIR}/lib)
    list (APPEND PETSC_LIB "${PETSC${PKG}_LIB}")
  endforeach ()
endif ()
''')
cmake.close()
cmakeok = False
if sys.version_info >= (2,5) and not petscconf.ISINSTALL and petscconf.BUILD_USING_CMAKE:
  import cmakegen
  try:
    cmakegen.main(slepcdir,petscdir,petscarch=petscconf.ARCH)
  except (OSError), e:
    log.Exit('ERROR: Generating CMakeLists.txt failed:\n' + str(e))
  import cmakeboot
  try:
    cmakeok = cmakeboot.main(slepcdir,petscdir,petscarch=petscconf.ARCH,log=log)
  except (OSError), e:
    log.Exit('ERROR: Booting CMake in PETSC_ARCH failed:\n' + str(e))
  except (ImportError, KeyError), e:
    log.Exit('ERROR: Importing cmakeboot failed:\n' + str(e))
  # remove files created by PETSc's script
  for f in ['build.log','build.log.bkp','RDict.log']:
    try: os.remove(f)
    except OSError: pass
if cmakeok:
  slepcvars.write('SLEPC_BUILD_USING_CMAKE = 1\n')

# Finish with configuration files
slepcvars.close()
slepcrules.close()
slepcconf.write('#endif\n')
slepcconf.close()
shutil.rmtree(tmpdir)

# Print summary
log.Println('')
log.Println('='*79)
log.Println('SLEPc Configuration')
log.Println('='*79)
log.Println('')
log.Println('SLEPc directory:')
log.Println(' '+slepcdir)
if archdir != prefixdir:
  log.Println('SLEPc prefix directory:')
  log.Println(' '+prefixdir)  
log.Println('PETSc directory:')
log.Println(' '+petscdir)
log.Println('Architecture "'+petscconf.ARCH+'" with '+petscconf.PRECISION+' precision '+petscconf.SCALAR+' numbers')
if havearpack:
  log.Println('ARPACK library flags:')
  log.Println(' '+str.join(' ',arpacklibs))
if haveblzpack:
  log.Println('BLZPACK library flags:')
  log.Println(' '+str.join(' ',blzpacklibs))
if havetrlan:
  log.Println('TRLAN library flags:')
  log.Println(' '+str.join(' ',trlanlibs))
if haveprimme:
  log.Println('PRIMME library flags:')
  log.Println(' '+str.join(' ',primmelibs))
if haveblopex:
  log.Println('BLOPEX library flags:')
  log.Println(' '+str.join(' ',blopexlibs))
if missing:
  log.Println('LAPACK missing functions:')
  log.Print('  ')
  for i in missing: log.Print(i)
  log.Println('')
  log.Println('')
  log.Println('WARNING: Some SLEPc functionality will not be available')
  log.Println('PLEASE reconfigure and recompile PETSc with a full LAPACK implementation')
print
print 'xxx'+'='*73+'xxx'
if cmakeok: buildtype = 'cmake'
else: buildtype = 'legacy'
print ' Configure stage complete. Now build the SLEPc library with ('+buildtype+' build):'
print '   make SLEPC_DIR=$PWD PETSC_DIR='+petscdir+' PETSC_ARCH='+petscconf.ARCH
print 'xxx'+'='*73+'xxx'
print
