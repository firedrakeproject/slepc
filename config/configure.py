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

import os, sys, time, tempfile, shutil

# Use en_US as language so that compiler messages are in English
if 'LC_LOCAL' in os.environ and os.environ['LC_LOCAL'] != '' and os.environ['LC_LOCAL'] != 'en_US' and os.environ['LC_LOCAL']!= 'en_US.UTF-8': os.environ['LC_LOCAL'] = 'en_US.UTF-8'
if 'LANG' in os.environ and os.environ['LANG'] != '' and os.environ['LANG'] != 'en_US' and os.environ['LANG'] != 'en_US.UTF-8': os.environ['LANG'] = 'en_US.UTF-8'

# should be run from the toplevel
configDir = os.path.abspath('config')
if not os.path.isdir(configDir):
  raise RuntimeError('Run configure from $SLEPC_DIR, not '+os.path.abspath('.'))
sys.path.insert(0, configDir)
sys.path.insert(0, os.path.join(configDir,'packages'))

if not hasattr(sys, 'version_info') or not sys.version_info[0] == 2 or not sys.version_info[1] >= 4:
  print '*****  You must have Python2 version 2.4 or higher to run ./configure.py   ******'
  print '*           Python is easy to install for end users or sys-admin.               *'
  print '*                   http://www.python.org/download/                             *'
  print '*                                                                               *'
  print '*            You CANNOT configure SLEPc without Python                          *'
  print '*********************************************************************************'
  sys.exit(4)

import argdb, log
argdb = argdb.ArgDB(sys.argv)
log = log.Log()

import petsc, arpack, blzpack, trlan, feast, primme, blopex, sowing
petsc = petsc.Petsc(log)
arpack = arpack.Arpack(argdb,log)
blzpack = blzpack.Blzpack(argdb,log)
trlan = trlan.Trlan(argdb,log)
primme = primme.Primme(argdb,log)
feast = feast.Feast(argdb,log)
blopex = blopex.Blopex(argdb,log)
sowing = sowing.Sowing(argdb,log)
doclean = argdb.PopBool('with-clean')
prefixdir = argdb.PopPath('prefix')[0]
datafilespath = argdb.PopPath('DATAFILESPATH')[0]

if argdb.PopHelp():
  print 'SLEPc Configure Help'
  print '-'*80
  print 'SLEPc:'
  print '  --with-clean=<bool>              : Delete prior build files including externalpackages'
  print '  --prefix=<dir>                   : Specify location to install SLEPc (e.g., /usr/local)'
  print '  --DATAFILESPATH=<dir>            : Specify location of datafiles (for SLEPc developers)'
  for pk in [ arpack, blzpack, trlan, primme, feast, blopex, sowing ]:
    pk.ShowHelp()
  sys.exit(0)

argdb.ErrorIfNotEmpty()

external = arpack.havepackage or blzpack.havepackage or trlan.havepackage or primme.havepackage or feast.havepackage or blopex.downloadpackage
prefixinstall = not prefixdir==''

# Check if enviroment is ok
print 'Checking environment...'
if 'SLEPC_DIR' in os.environ:
  slepcdir = os.environ['SLEPC_DIR']
  if not os.path.exists(slepcdir) or not os.path.exists(os.path.join(slepcdir,'config')):
    sys.exit('ERROR: SLEPC_DIR enviroment variable is not valid')
  if os.path.realpath(os.getcwd()) != os.path.realpath(slepcdir):
    sys.exit('ERROR: SLEPC_DIR is not the current directory')
else:
  slepcdir = os.getcwd()
  if not os.path.exists(os.path.join(slepcdir,'config')):
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
import petscversion, slepcversion
petscversion.Load(petscdir)
slepcversion.Load(slepcdir)
if petscversion.VERSION < slepcversion.VERSION:
  sys.exit('ERROR: This SLEPc version is not compatible with PETSc version '+petscversion.VERSION)

# Check some information about PETSc configuration
import petscconf
petscconf.Load(petscdir)
if not petscconf.PRECISION in ['double','single','__float128']:
  sys.exit('ERROR: This SLEPc version does not work with '+petscconf.PRECISION+' precision')
if prefixinstall and not petscconf.ISINSTALL:
  sys.exit('ERROR: SLEPc cannot be configured for non-source installation if PETSc is not configured in the same way.')

# Check for empty PETSC_ARCH
archname = petscconf.ARCH
emptyarch = 1
if 'PETSC_ARCH' in os.environ and os.environ['PETSC_ARCH']: emptyarch = 0
if emptyarch:
  archname = 'installed-' + petscconf.ARCH
  globconfdir = os.path.join(slepcdir,'lib','slepc-conf')
  try:
    globconf = open(os.path.join(globconfdir,'slepcvariables'),'w')
    globconf.write('SLEPC_DIR = ' + slepcdir +'\n')
    globconf.write('PETSC_ARCH = ' + archname + '\n')
    globconf.close()
  except:
    sys.exit('ERROR: cannot create configuration file in ' + globconfdir)
archdir = os.path.join(slepcdir,archname)

# Clean previous configuration if needed
if os.path.exists(archdir):
  try:
    f = open(os.path.join(archdir,'lib','slepc-conf','slepcvariables'),"r")
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
incdir = os.path.join(archdir,'include')
if not os.path.exists(incdir):
  try:
    os.mkdir(incdir)
  except:
    sys.exit('ERROR: cannot create include directory ' + incdir)
libdir = os.path.join(archdir,'lib')
if not os.path.exists(libdir):
  try:
    os.mkdir(libdir)
  except:
    sys.exit('ERROR: cannot create lib directory ' + libdir)
confdir = os.path.join(libdir,'slepc-conf')
if not os.path.exists(confdir):
  try:
    os.mkdir(confdir)
  except:
    sys.exit('ERROR: cannot create configuration directory ' + confdir)
modulesbasedir = os.path.join(confdir,'modules')
if not os.path.exists(modulesbasedir):
  try:
    os.mkdir(modulesbasedir)
  except:
    sys.exit('ERROR: cannot create modules base directory ' + modulesbasedir)
modulesdir = os.path.join(modulesbasedir,'slepc')
if not os.path.exists(modulesdir):
  try:
    os.mkdir(modulesdir)
  except:
    sys.exit('ERROR: cannot create modules directory ' + modulesdir)
pkgconfigdir = os.path.join(libdir,'pkgconfig')
if not os.path.exists(pkgconfigdir):
  try:
    os.mkdir(pkgconfigdir)
  except:
    sys.exit('ERROR: cannot create pkgconfig directory ' + pkgconfigdir)
try:
  slepcvars = open(os.path.join(confdir,'slepcvariables'),'w')
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
  slepcrules = open(os.path.join(confdir,'slepcrules'),'w')
except:
  sys.exit('ERROR: cannot create rules file in ' + confdir)
try:
  slepcconf = open(os.path.join(incdir,'slepcconf.h'),'w')
  slepcconf.write('#if !defined(__SLEPCCONF_H)\n')
  slepcconf.write('#define __SLEPCCONF_H\n\n')
  if slepcversion.ISREPO:
    slepcconf.write('#ifndef SLEPC_VERSION_GIT\n#define SLEPC_VERSION_GIT "' + slepcversion.GITREV + '"\n#endif\n\n')
    slepcconf.write('#ifndef SLEPC_VERSION_DATE_GIT\n#define SLEPC_VERSION_DATE_GIT "' + slepcversion.GITDATE + '"\n#endif\n\n')
  slepcconf.write('#ifndef SLEPC_LIB_DIR\n#define SLEPC_LIB_DIR "' + os.path.join(prefixdir,'lib') + '"\n#endif\n\n')
except:
  sys.exit('ERROR: cannot create configuration header in ' + confdir)
try:
  cmake = open(os.path.join(confdir,'SLEPcConfig.cmake'),'w')
except:
  sys.exit('ERROR: cannot create CMake configuration file in ' + confdir)
try:
  if archdir != prefixdir:
    modules = open(os.path.join(modulesdir,slepcversion.LVERSION),'w')
  else:
    modules = open(os.path.join(modulesdir,slepcversion.LVERSION+'-'+archname),'w')
except:
  sys.exit('ERROR: cannot create modules file in ' + modulesdir)
try:
  pkgconfig = open(os.path.join(pkgconfigdir,'SLEPc.pc'),'w')
except:
  sys.exit('ERROR: cannot create pkgconfig file in ' + pkgconfigdir)

# Create temporary directory and makefile for running tests
try:
  tmpdir = tempfile.mkdtemp(prefix='slepc-')
  if not os.path.isdir(tmpdir): os.mkdir(tmpdir)
except:
  sys.exit('ERROR: cannot create temporary directory')
try:
  makefile = open(os.path.join(tmpdir,'makefile'),'w')
  makefile.write('checklink: checklink.o chkopts\n')
  makefile.write('\t${CLINKER} -o checklink checklink.o ${TESTFLAGS} ${PETSC_KSP_LIB}\n')
  makefile.write('\t@${RM} -f checklink checklink.o\n')
  makefile.write('LOCDIR = ./\n')
  makefile.write('include '+os.path.join('${PETSC_DIR}','lib','petsc-conf','variables')+'\n')
  makefile.write('include '+os.path.join('${PETSC_DIR}','lib','petsc-conf','rules')+'\n')
  makefile.close()
except:
  sys.exit('ERROR: cannot create makefile in temporary directory')

# Open log file
log.Open(os.path.join(confdir,'configure.log'))
log.write('='*80)
log.write('Starting Configure Run at '+time.ctime(time.time()))
log.write('Configure Options: '+' '.join(sys.argv[1:]))
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
petsc.Check(tmpdir)
if not petsc.havepackage:
  log.Exit('ERROR: Unable to link with PETSc')

# Single library installation
if petscconf.SINGLELIB:
  slepcvars.write('SHLIBS = libslepc\n')
  slepcvars.write('LIBNAME = '+os.path.join('${INSTALL_LIB_DIR}','libslepc.${AR_LIB_SUFFIX}')+'\n')
  for module in ['SYS','MFN','EPS','SVD','PEP','NEP']:
    slepcvars.write('SLEPC_'+module+'_LIB = ${CC_LINKER_SLFLAG}${SLEPC_LIB_DIR} -L${SLEPC_LIB_DIR} -lslepc ${SLEPC_EXTERNAL_LIB} ${PETSC_KSP_LIB}\n')
  slepcvars.write('SLEPC_LIB = ${CC_LINKER_SLFLAG}${SLEPC_LIB_DIR} -L${SLEPC_LIB_DIR} -lslepc ${SLEPC_EXTERNAL_LIB} ${PETSC_KSP_LIB}\n')

# Check for external packages
if arpack.havepackage:
  arpack.Check(slepcconf,slepcvars,cmake,tmpdir)
if blzpack.havepackage:
  blzpack.Check(slepcconf,slepcvars,cmake,tmpdir)
if trlan.havepackage:
  trlan.Check(slepcconf,slepcvars,cmake,tmpdir)
if primme.havepackage:
  primme.Check(slepcconf,slepcvars,cmake,tmpdir)
if feast.havepackage:
  feast.Check(slepcconf,slepcvars,cmake,tmpdir)
if blopex.downloadpackage:
  blopex.Install(slepcconf,slepcvars,cmake,tmpdir,archdir)

# Check for missing LAPACK functions
import lapack
lapack = lapack.Lapack(log)
missing = lapack.Check(slepcconf,slepcvars,cmake,tmpdir)

# Download sowing if requested and make Fortran stubs if necessary
bfort = petscconf.BFORT
if sowing.downloadpackage:
  bfort = sowing.Install(archdir)

if slepcversion.ISREPO and hasattr(petscconf,'FC'):
  try:
    if not os.path.exists(bfort):
      bfort = os.path.join(archdir,'bin','bfort')
    if not os.path.exists(bfort):
      bfort = sowing.Install(archdir)
    sys.path.insert(0, os.path.abspath(os.path.join('bin','maint')))
    import generatefortranstubs
    generatefortranstubs.main(slepcdir,bfort,os.getcwd(),0)
    generatefortranstubs.processf90interfaces(slepcdir,0)
  except AttributeError:
    sys.exit('ERROR: cannot generate Fortran stubs; try configuring PETSc with --download-sowing or use a mercurial version of PETSc')

if bfort != petscconf.BFORT:
  slepcvars.write('BFORT = '+bfort+'\n')

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
pkgconfig.write('Cflags: -I' + os.path.join(prefixdir,'include'))
if not prefixinstall:
  pkgconfig.write(' -I' + os.path.join(slepcdir,'include'))
pkgconfig.write('\nLibs: -L%s -lslepc\n' % os.path.join(prefixdir,'lib'))

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
  log.Println('Architecture "'+archname+'" with '+petscconf.PRECISION+' precision '+petscconf.SCALAR+' numbers')
if arpack.havepackage:
  log.Println('ARPACK library flags:')
  log.Println(' '+' '.join(arpack.packagelibs))
if blzpack.havepackage:
  log.Println('BLZPACK library flags:')
  log.Println(' '+' '.join(blzpack.packagelibs))
if trlan.havepackage:
  log.Println('TRLAN library flags:')
  log.Println(' '+' '.join(trlan.packagelibs))
if primme.havepackage:
  log.Println('PRIMME library flags:')
  log.Println(' '+' '.join(primme.packagelibs))
if feast.havepackage:
  log.Println('FEAST library flags:')
  log.Println(' '+' '.join(feast.packagelibs))
if blopex.havepackage:
  log.Println('BLOPEX library flags:')
  log.Println(' '+' '.join(blopex.packagelibs))
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
if emptyarch:
  print '   make SLEPC_DIR=$PWD PETSC_DIR='+petscdir
else:
  print '   make SLEPC_DIR=$PWD PETSC_DIR='+petscdir+' PETSC_ARCH='+archname
print 'xxx'+'='*73+'xxx'
print
