#!/usr/bin/env python
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
import time
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
import slepcversion
import petscconf
import log
import check
import arpack
import blzpack
import trlan
import feast
import lapack
import primme
import blopex

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
  if name.startswith('--with'):
    if name.find('=') == -1: sys.argv[l] += '=1'

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
havefeast = 0
feastdir = ''
feastlibs = []
getblopex = 0
haveblopex = 0
blopexurl = ''
doclean = 0
prefixdir = ''
datafilespath = ''

for i in sys.argv[1:]:
  if   i.startswith('--with-arpack-dir='):
    arpackdir = i.split('=')[1]
    havearpack = 1
  elif i.startswith('--with-arpack-flags='):
    arpacklibs = i.split('=')[1].split(',')
    havearpack = 1
  elif i.startswith('--with-arpack='):
    havearpack = not i.endswith('=0')
  elif i.startswith('--with-blzpack-dir='):
    blzpackdir = i.split('=')[1]
    haveblzpack = 1
  elif i.startswith('--with-blzpack-flags='):
    blzpacklibs = i.split('=')[1].split(',')
    haveblzpack = 1
  elif i.startswith('--with-blzpack='):
    haveblzpack = not i.endswith('=0')
  elif i.startswith('--with-trlan-dir='):
    trlandir = i.split('=')[1]
    havetrlan = 1
  elif i.startswith('--with-trlan-flags='):
    trlanlibs = i.split('=')[1].split(',')
    havetrlan = 1
  elif i.startswith('--with-trlan='):
    havetrlan = not i.endswith('=0')
  elif i.startswith('--with-primme-dir='):
    primmedir = i.split('=')[1]
    haveprimme = 1
  elif i.startswith('--with-primme-flags='):
    primmelibs = i.split('=')[1].split(',')
    haveprimme = 1
  elif i.startswith('--with-primme='):
    haveprimme = not i.endswith('=0')
  elif i.startswith('--with-feast-dir='):
    feastdir = i.split('=')[1]
    havefeast = 1
  elif i.startswith('--with-feast-flags='):
    feastlibs = i.split('=')[1].split(',')
    havefeast = 1
  elif i.startswith('--with-feast='):
    havefeast = not i.endswith('=0')
  elif i.startswith('--download-blopex'):
    getblopex = not i.endswith('=0')
    try: blopexurl = i.split('=')[1]
    except IndexError: pass
  elif i.startswith('--with-clean='):
    doclean = not i.endswith('=0')
  elif i.startswith('--prefix='):
    prefixdir = i.split('=')[1]
  elif i.startswith('--DATAFILESPATH='):
    datafilespath = i.split('=')[1]
  elif i.startswith('--h') or i.startswith('-h') or i.startswith('-?'):
    print 'SLEPc Configure Help'
    print '-'*80
    print 'SLEPc:'
    print '  --with-clean=<bool>              : Delete prior build files including externalpackages'
    print '  --prefix=<dir>                   : Specify location to install SLEPc (e.g., /usr/local)'
    print '  --DATAFILESPATH=<dir>            : Specify location of datafiles (for SLEPc developers)'
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
    print 'FEAST:'
    print '  --with-feast                     : Indicate if you wish to test for FEAST'
    print '  --with-feast-dir=<dir>           : Indicate the directory for FEAST libraries'
    print '  --with-feast-flags=<flags>       : Indicate comma-separated flags for linking FEAST'
    print 'BLOPEX:'
    print '  --download-blopex                : Download and install BLOPEX in SLEPc directory'
    sys.exit(0)
  else:
    sys.exit('ERROR: Invalid argument ' + i +'. Use -h for help')

external = havearpack or haveblzpack or havetrlan or haveprimme or havefeast or getblopex
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
slepcversion.Load(slepcdir)
if petscversion.VERSION < slepcversion.VERSION:
  sys.exit('ERROR: This SLEPc version is not compatible with PETSc version '+petscversion.VERSION)

# Check some information about PETSc configuration
petscconf.Load(petscdir)
if not petscconf.PRECISION in ['double','single','__float128']:
  sys.exit('ERROR: This SLEPc version does not work with '+petscconf.PRECISION+' precision')
if prefixinstall and not petscconf.ISINSTALL:
  sys.exit('ERROR: SLEPc cannot be configured for non-source installation if PETSc is not configured in the same way.')

# Check for empty PETSC_ARCH
archdir = os.sep.join([slepcdir,petscconf.ARCH])
emptyarch = 1
if 'PETSC_ARCH' in os.environ and os.environ['PETSC_ARCH']: emptyarch = 0
if emptyarch:
  globconfdir = os.sep.join([slepcdir,'conf'])
  try:
    globconf = open(os.sep.join([globconfdir,'slepcvariables']),'w')
    globconf.write('SLEPC_DIR = ' + slepcdir +'\n')
    globconf.write('PETSC_ARCH = ' + petscconf.ARCH +'\n')
    globconf.close()
  except:
    sys.exit('ERROR: cannot create configuration file in ' + globconfdir)

# Clean previous configuration if needed
if os.path.exists(archdir):
  try:
    f = open(os.sep.join([archdir,'conf/slepcvariables']),"r")
    searchlines = f.readlines()
    f.close()
    found = 0
    for library in ['ARPACK','BLZPACK','TRLAN','PRIMME','FEAST','BLOPEX']:
      if library in ''.join(searchlines):
        found = 1
    if found and not external:
      print 'WARNING: forcing --with-clean=1 because previous configuration had external packages'
      doclean = 1
  except: pass
  if doclean:
    try:
      shutil.rmtree(archdir)
    except:
      sys.exit('ERROR: cannot remove existing directory ' + archdir)

# Create architecture directory and configuration files
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
modulesdir = os.sep.join([libdir,'modules'])
if not os.path.exists(modulesdir):
  try:
    os.mkdir(modulesdir)
  except:
    sys.exit('ERROR: cannot create modules directory ' + modulesdir)
pkgconfigdir = os.sep.join([libdir,'pkgconfig'])
if not os.path.exists(pkgconfigdir):
  try:
    os.mkdir(pkgconfigdir)
  except:
    sys.exit('ERROR: cannot create pkgconfig directory ' + pkgconfigdir)
try:
  slepcvars = open(os.sep.join([confdir,'slepcvariables']),'w')
  if not prefixdir:
    prefixdir = archdir
  slepcvars.write('SLEPC_DESTDIR = ' + prefixdir +'\n')
  if emptyarch:
    slepcvars.write('INSTALLED_PETSC = 1\n')
  testruns = set(petscconf.TEST_RUNS.split())
  testruns = testruns.intersection(set(['C','F90','Fortran','C_Complex','Fortran_Complex','C_NoComplex','Fortran_NoComplex']))
  if petscconf.PRECISION != '__float128':
    testruns = testruns.union(set(['C_NoF128']))
  if datafilespath:
    slepcvars.write('DATAFILESPATH = ' + datafilespath +'\n')
    testruns = testruns.union(set(['DATAFILESPATH']))
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
  if slepcversion.ISREPO:
    slepcconf.write('#ifndef SLEPC_VERSION_GIT\n#define SLEPC_VERSION_GIT "' + slepcversion.GITREV + '"\n#endif\n\n')
    slepcconf.write('#ifndef SLEPC_VERSION_DATE_GIT\n#define SLEPC_VERSION_DATE_GIT "' + slepcversion.GITDATE + '"\n#endif\n\n')
  slepcconf.write('#ifndef SLEPC_LIB_DIR\n#define SLEPC_LIB_DIR "' + prefixdir + '/lib"\n#endif\n\n')
except:
  sys.exit('ERROR: cannot create configuration header in ' + confdir)
try:
  cmake = open(os.sep.join([confdir,'SLEPcConfig.cmake']),'w')
except:
  sys.exit('ERROR: cannot create CMake configuration file in ' + confdir)
try:
  if archdir != prefixdir:
    modules = open(os.sep.join([modulesdir,slepcversion.LVERSION]),'w')
  else:
    modules = open(os.sep.join([modulesdir,slepcversion.LVERSION+'-'+petscconf.ARCH]),'w')
except:
  sys.exit('ERROR: cannot create modules file in ' + modulesdir)
try:
  pkgconfig = open(os.sep.join([pkgconfigdir,'SLEPc.pc']),'w')
except:
  sys.exit('ERROR: cannot create pkgconfig file in ' + pkgconfigdir)

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
log.write('PETSc version: ' + petscversion.LVERSION)
if not emptyarch:
  log.write('PETSc architecture: ' + petscconf.ARCH)
log.write('SLEPc source directory: ' + slepcdir)
log.write('SLEPc install directory: ' + prefixdir)
log.write('SLEPc version: ' + slepcversion.LVERSION)
log.write('='*80)

# Check if PETSc is working
log.Println('Checking PETSc installation...')
if petscversion.VERSION > slepcversion.VERSION:
  log.Println('WARNING: PETSc version '+petscversion.VERSION+' is newer than SLEPc version '+slepcversion.VERSION)
if petscversion.RELEASE != slepcversion.RELEASE:
  sys.exit('ERROR: Cannot mix release and development versions of SLEPc and PETSc')
if petscconf.ISINSTALL:
  if os.path.realpath(petscconf.DESTDIR) != os.path.realpath(petscdir):
    log.Println('WARNING: PETSC_DIR does not point to PETSc installation path')
if not check.Link(tmpdir,[],[],[]):
  log.Exit('ERROR: Unable to link with PETSc')

# Single library installation
if petscconf.SINGLELIB:
  slepcvars.write('SHLIBS = libslepc\n')
  slepcvars.write('LIBNAME = ${INSTALL_LIB_DIR}/libslepc.${AR_LIB_SUFFIX}\n')
  for module in ['SYS','MFN','EPS','SVD','PEP','NEP']:
    slepcvars.write('SLEPC_'+module+'_LIB = ${CC_LINKER_SLFLAG}${SLEPC_LIB_DIR} -L${SLEPC_LIB_DIR} -lslepc ${SLEPC_EXTERNAL_LIB} ${PETSC_KSP_LIB}\n')
  slepcvars.write('SLEPC_LIB = ${CC_LINKER_SLFLAG}${SLEPC_LIB_DIR} -L${SLEPC_LIB_DIR} -lslepc ${SLEPC_EXTERNAL_LIB} ${PETSC_KSP_LIB}\n')

# Check for external packages
if havearpack:
  arpacklibs = arpack.Check(slepcconf,slepcvars,cmake,tmpdir,arpackdir,arpacklibs)
if haveblzpack:
  blzpacklibs = blzpack.Check(slepcconf,slepcvars,cmake,tmpdir,blzpackdir,blzpacklibs)
if havetrlan:
  trlanlibs = trlan.Check(slepcconf,slepcvars,cmake,tmpdir,trlandir,trlanlibs)
if haveprimme:
  primmelibs = primme.Check(slepcconf,slepcvars,cmake,tmpdir,primmedir,primmelibs)
if havefeast:
  feastlibs = feast.Check(slepcconf,slepcvars,cmake,tmpdir,feastdir,feastlibs)
if getblopex:
  blopexlibs = blopex.Install(slepcconf,slepcvars,cmake,tmpdir,blopexurl,archdir)
  haveblopex = 1

# Check for missing LAPACK functions
missing = lapack.Check(slepcconf,slepcvars,cmake,tmpdir)

# Make Fortran stubs if necessary
if slepcversion.ISREPO and hasattr(petscconf,'FC'):
  try:
    import generatefortranstubs
    generatefortranstubs.main(slepcdir,petscconf.BFORT,os.getcwd(),0)
    generatefortranstubs.processf90interfaces(slepcdir,0)
    for f in os.listdir(os.sep.join([slepcdir,'include/finclude/ftn-auto'])):
      if '-tmpdir' in f: shutil.rmtree(os.sep.join([slepcdir,'include/finclude/ftn-auto/',f]))
      if 'petsc' in f: os.remove(os.sep.join([slepcdir,'include/finclude/ftn-auto/',f]))
  except AttributeError:
    sys.exit('ERROR: cannot generate Fortran stubs; try configuring PETSc with --download-sowing or use a mercurial version of PETSc')

# CMake stuff
cmake.write('set (SLEPC_PACKAGE_LIBS "${ARPACK_LIB}" "${BLZPACK_LIB}" "${TRLAN_LIB}" "${PRIMME_LIB}" "${FEAST_LIB}" "${BLOPEX_LIB}" )\n')
cmake.write('set (SLEPC_PACKAGE_INCLUDES "${PRIMME_INCLUDE}")\n')
cmake.write('find_library (PETSC_LIB petsc HINTS ${PETSc_BINARY_DIR}/lib )\n')
cmake.write('''
if (NOT PETSC_LIB) # Interpret missing libpetsc to mean that PETSc was built --with-single-library=0
  set (PETSC_LIB "")
  foreach (pkg sys vec mat dm ksp snes ts tao)
    string (TOUPPER ${pkg} PKG)
    find_library (PETSC${PKG}_LIB "petsc${pkg}" HINTS ${PETSc_BINARY_DIR}/lib)
    list (APPEND PETSC_LIB "${PETSC${PKG}_LIB}")
  endforeach ()
endif ()
''')
cmake.close()
cmakeok = False
if sys.version_info >= (2,5) and not petscconf.ISINSTALL and petscconf.BUILD_USING_CMAKE:
  import cmakegen
  try:
    cmakegen.main(slepcdir,petscdir,petscdestdir=petscconf.DESTDIR)
  except (OSError), e:
    log.Exit('ERROR: Generating CMakeLists.txt failed:\n' + str(e))
  import cmakeboot
  try:
    cmakeok = cmakeboot.main(slepcdir,petscdir,log=log)
  except (OSError), e:
    log.Exit('ERROR: Booting CMake in PETSC_ARCH failed:\n' + str(e))
  except (ImportError, KeyError), e:
    log.Exit('ERROR: Importing cmakeboot failed:\n' + str(e))
  except (AttributeError), e:
    log.Println('xxx'+'='*73+'xxx')
    log.Println('WARNING: CMake builds are not available (initialization failed)')
    log.Println('You can ignore this warning (use default build), or try reconfiguring PETSc')
    log.Println('xxx'+'='*73+'xxx')
  # remove files created by PETSc's script
  for f in ['build.log','build.log.bkp','RDict.log']:
    try: os.remove(f)
    except OSError: pass
if cmakeok:
  slepcvars.write('SLEPC_BUILD_USING_CMAKE = 1\n')

# Modules file
modules.write('#%Module\n\n')
modules.write('proc ModulesHelp { } {\n')
modules.write('    puts stderr "This module sets the path and environment variables for slepc-%s"\n' % slepcversion.LVERSION)
modules.write('    puts stderr "     see http://slepc.upv.es/ for more information"\n')
modules.write('    puts stderr ""\n}\n')
modules.write('module-whatis "SLEPc - Scalable Library for Eigenvalue Problem Computations"\n\n')
modules.write('module load petsc\n')
if prefixinstall:
  modules.write('set slepc_dir %s\n' % prefixdir)
else:
  modules.write('set slepc_dir %s\n' % slepcdir)
modules.write('setenv SLEPC_DIR $slepc_dir\n')

# pkg-config file
pkgconfig.write('Name: SLEPc, the Scalable Library for Eigenvalue Problem Computations\n')
pkgconfig.write('Description: A parallel library to compute eigenvalues and eigenvectors of large, sparse matrices with iterative methods. It is based on PETSc.\n')
pkgconfig.write('Version: %s\n' % slepcversion.LVERSION)
pkgconfig.write('Requires: PETSc = %s\n' % petscversion.LVERSION)
pkgconfig.write('Cflags: -I%s/include' % prefixdir)
if not prefixinstall:
  pkgconfig.write(' -I%s/include' % slepcdir)
pkgconfig.write('\nLibs: -L%s/lib -lslepc\n' % prefixdir)

# Finish with configuration files
slepcvars.close()
slepcrules.close()
slepcconf.write('#endif\n')
slepcconf.close()
modules.close()
pkgconfig.close()
shutil.rmtree(tmpdir)

# Print summary
log.Println('')
log.Println('='*79)
log.Println('SLEPc Configuration')
log.Println('='*79)
log.Println('')
log.Println('SLEPc directory:')
log.Println(' '+slepcdir)
if slepcversion.ISREPO:
  log.Println('  It is a git repository on branch: '+slepcversion.BRANCH)
if archdir != prefixdir:
  log.Println('SLEPc prefix directory:')
  log.Println(' '+prefixdir)
log.Println('PETSc directory:')
log.Println(' '+petscdir)
if petscversion.ISREPO:
  log.Println('  It is a git repository on branch: '+petscversion.BRANCH)
if petscversion.ISREPO and slepcversion.ISREPO:
  if petscversion.BRANCH!='maint' and slepcversion.BRANCH!='maint':
    try:
      import dateutil.parser
      import datetime
      petscdate = dateutil.parser.parse(petscversion.GITDATE)
      slepcdate = dateutil.parser.parse(slepcversion.GITDATE)
      if abs(petscdate-slepcdate)>datetime.timedelta(days=30):
        log.Println('xxx'+'='*73+'xxx')
        log.Println('WARNING: your PETSc and SLEPc repos may not be in sync (more than 30 days apart)')
        log.Println('xxx'+'='*73+'xxx')
    except ImportError: pass
if emptyarch and archdir != prefixdir:
  log.Println('Prefix install with '+petscconf.PRECISION+' precision '+petscconf.SCALAR+' numbers')
else:
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
if havefeast:
  log.Println('FEAST library flags:')
  log.Println(' '+str.join(' ',feastlibs))
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
if petscconf.MAKE_IS_GNUMAKE: buildtype = 'gnumake'
elif cmakeok: buildtype = 'cmake'
else: buildtype = 'legacy'
print ' Configure stage complete. Now build the SLEPc library with ('+buildtype+' build):'
if emptyarch and archdir != prefixdir:
  print '   make SLEPC_DIR=$PWD PETSC_DIR='+petscdir
else:
  print '   make SLEPC_DIR=$PWD PETSC_DIR='+petscdir+' PETSC_ARCH='+petscconf.ARCH
print 'xxx'+'='*73+'xxx'
print
