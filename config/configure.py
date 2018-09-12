#!/usr/bin/env python
#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

from __future__ import print_function
import os, sys, time, shutil

def AddDefine(conffile,name,value,prefix='SLEPC_'):
  conffile.write('#ifndef '+prefix+name+'\n#define '+prefix+name+' "'+value+'"\n#endif\n\n')

def CreateFile(basedir,fname,log):
  ''' Create file basedir/fname and return path string '''
  newfile = os.path.join(basedir,fname)
  try:
    newfile = open(newfile,'w')
  except:
    log.Exit('ERROR: Cannot create '+fname+' file in '+basedir)
  return newfile

def CreateDir(basedir,dirname,log):
  ''' Create directory basedir/dirname and return path string '''
  newdir = os.path.join(basedir,dirname)
  if not os.path.exists(newdir):
    try:
      os.mkdir(newdir)
    except:
      log.Exit('ERROR: Cannot create '+dirname+' directory: '+newdir)
  return newdir

def CreateDirTwo(basedir,dir1,dir2,log):
  ''' Create directory basedir/dir1/dir2 and return path string '''
  newbasedir = os.path.join(basedir,dir1)
  if not os.path.exists(newbasedir):
    try:
      os.mkdir(newbasedir)
    except:
      log.Exit('ERROR: Cannot create '+dir1+' directory: '+newbasedir)
  newdir = os.path.join(newbasedir,dir2)
  if not os.path.exists(newdir):
    try:
      os.mkdir(newdir)
    except:
      log.Exit('ERROR: Cannot create '+dirname+' directory: '+newdir)
  return newdir

def CreateDirTest(basedir,dirname,log):
  ''' Create directory, return path string and flag indicating if already existed '''
  newdir = os.path.join(basedir,dirname)
  if not os.path.exists(newdir):
    existed = False
    try:
      os.mkdir(newdir)
    except:
      log.Exit('ERROR: Cannot create '+dirname+' directory: '+newdir)
  else:
    existed = True
  return newdir, existed

def WriteModulesFile(modules,version,sdir):
  ''' Write the contents of the Modules file '''
  modules.write('#%Module\n\n')
  modules.write('proc ModulesHelp { } {\n')
  modules.write('    puts stderr "This module sets the path and environment variables for slepc-%s"\n' % version)
  modules.write('    puts stderr "     see http://slepc.upv.es/ for more information"\n')
  modules.write('    puts stderr ""\n}\n')
  modules.write('module-whatis "SLEPc - Scalable Library for Eigenvalue Problem Computations"\n\n')
  modules.write('module load petsc\n')
  modules.write('set slepc_dir %s\n' % sdir)
  modules.write('setenv SLEPC_DIR $slepc_dir\n')

def WritePkgconfigFile(pkgconfig,version,pversion,sdir,isinstall,prefixdir,singlelib):
  ''' Write the contents of the pkg-config file '''
  pkgconfig.write('prefix=%s\n' % prefixdir)
  pkgconfig.write('exec_prefix=${prefix}\n')
  pkgconfig.write('includedir=${prefix}/include\n')
  pkgconfig.write('libdir=${prefix}/lib\n\n')
  pkgconfig.write('Name: SLEPc\n')
  pkgconfig.write('Description: the Scalable Library for Eigenvalue Problem Computations\n')
  pkgconfig.write('Version: %s\n' % version)
  pkgconfig.write('Requires: PETSc >= %s\n' % pversion)
  pkgconfig.write('Cflags: -I${includedir}')
  if not isinstall:
    pkgconfig.write(' -I'+os.path.join(sdir,'include'))
  pkgconfig.write('\nLibs:')
  if singlelib:
    pkgconfig.write(' -L${libdir} -lslepc\n')
  else:
    pkgconfig.write(' -L${libdir} -lslepcnep -lslepcpep -lslepcsvd -lslepceps -lslepcmfn -lslepclme -lslepcsys\n')

def WriteCMakeConfigFile(cmakeconf):
  ''' Write the contents of the CMake configuration file '''
  cmakeconf.write('''
set (SLEPC_PACKAGE_LIBS "${ARPACK_LIB}" "${BLZPACK_LIB}" "${TRLAN_LIB}" "${PRIMME_LIB}" "${FEAST_LIB}" "${BLOPEX_LIB}" "${SLICOT_LIB}" )
set (SLEPC_PACKAGE_INCLUDES "${PRIMME_INCLUDE}")
find_library (PETSC_LIB petsc HINTS ${PETSc_BINARY_DIR}/lib )
if (NOT PETSC_LIB) # Interpret missing libpetsc to mean that PETSc was built --with-single-library=0
  set (PETSC_LIB "")
  foreach (pkg sys vec mat dm ksp snes ts tao)
    string (TOUPPER ${pkg} PKG)
    find_library (PETSC${PKG}_LIB "petsc${pkg}" HINTS ${PETSc_BINARY_DIR}/lib)
    list (APPEND PETSC_LIB "${PETSC${PKG}_LIB}")
  endforeach ()
endif ()
''')

def WriteReconfigScript(reconfig,slepcdir,usedargs):
  ''' Write the contents of the reconfigure script '''
  reconfig.write('#!/usr/bin/env python\n\n')
  reconfig.write('import os, sys\n')
  if usedargs:
    reconfig.write('sys.argv.extend(\''+usedargs+'\'.split())\n')
  reconfig.write('execfile(os.path.join(\''+slepcdir+'\',\'config\',\'configure.py\'))\n')

# Use en_US as language so that compiler messages are in English
if 'LC_LOCAL' in os.environ and os.environ['LC_LOCAL'] != '' and os.environ['LC_LOCAL'] != 'en_US' and os.environ['LC_LOCAL']!= 'en_US.UTF-8': os.environ['LC_LOCAL'] = 'en_US.UTF-8'
if 'LANG' in os.environ and os.environ['LANG'] != '' and os.environ['LANG'] != 'en_US' and os.environ['LANG'] != 'en_US.UTF-8': os.environ['LANG'] = 'en_US.UTF-8'

# Check python version
if not hasattr(sys, 'version_info') or not sys.version_info[0] == 2 or not sys.version_info[1] >= 6:
  print('*******************************************************************************')
  print('*       Python2 version 2.6 or higher is required to run ./configure          *')
  print('*          Try: "python2.7 ./configure" or "python2.6 ./configure"            *')
  print('*******************************************************************************')
  sys.exit(4)

# Set python path
configdir = os.path.abspath('config')
if not os.path.isdir(configdir):
  sys.exit('ERROR: Run configure from $SLEPC_DIR, not '+os.path.abspath('.'))
sys.path.insert(0,configdir)
sys.path.insert(0,os.path.join(configdir,'packages'))

# Load auxiliary classes
import argdb, log
argdb = argdb.ArgDB(sys.argv)
log   = log.Log()

# Load classes for packages and process command-line options
import slepc, petsc, arpack, blzpack, trlan, feast, primme, blopex, sowing, lapack, slicot
slepc   = slepc.SLEPc(argdb,log)
petsc   = petsc.PETSc(argdb,log)
arpack  = arpack.Arpack(argdb,log)
blopex  = blopex.Blopex(argdb,log)
blzpack = blzpack.Blzpack(argdb,log)
feast   = feast.Feast(argdb,log)
primme  = primme.Primme(argdb,log)
trlan   = trlan.Trlan(argdb,log)
sowing  = sowing.Sowing(argdb,log)
lapack  = lapack.Lapack(argdb,log)
slicot  = slicot.Slicot(argdb,log)

externalpackages = [arpack, blopex, blzpack, feast, primme, slicot, trlan]
optionspackages  = [slepc, arpack, blopex, blzpack, feast, primme, slicot, trlan, sowing]
checkpackages    = [arpack, blopex, blzpack, feast, primme, slicot, trlan, lapack]

# Print help if requested and check for wrong command-line options
if argdb.PopHelp():
  print('SLEPc Configure Help')
  print('-'*80)
  for pkg in optionspackages:
    pkg.ShowHelp()
  sys.exit(0)
argdb.ErrorIfNotEmpty()

# Check enviroment and PETSc version
print('Checking environment...', end=' ')
petsc.InitDir(slepc.prefixdir)
slepc.InitDir()
petsc.LoadVersion()
slepc.LoadVersion()
if petsc.nversion < slepc.nversion:
  sys.exit('ERROR: This SLEPc version is not compatible with PETSc version '+petsc.version)

# Check some information about PETSc configuration
petsc.LoadConf()
if not petsc.precision in ['double','single','__float128']:
  sys.exit('ERROR: This SLEPc version does not work with '+petsc.precision+' precision')

# Check for empty PETSC_ARCH
emptyarch = not ('PETSC_ARCH' in os.environ and os.environ['PETSC_ARCH'])
if emptyarch:
  pseudoarch = 'arch-' + sys.platform.replace('cygwin','mswin')+ '-' + petsc.language
  if petsc.debug:
    pseudoarch += '-debug'
  else:
    pseudoarch += '-opt'
  if not 'real' in petsc.scalar:
    pseudoarch += '-' + petsc.scalar
  archname = 'installed-'+pseudoarch
else:
  archname = petsc.arch

# Create directories for configuration files
archdir, archdirexisted = CreateDirTest(slepc.dir,archname,log)
libdir  = CreateDir(archdir,'lib',log)
confdir = CreateDirTwo(libdir,'slepc','conf',log)

# Open log file
log.Open(os.path.join(confdir,'configure.log'))
log.write('='*80)
log.write('Starting Configure Run at '+time.ctime(time.time()))
log.write('Configure Options: '+' '.join(sys.argv[1:]))
log.write('Working directory: '+os.getcwd())
log.write('Python version:\n'+sys.version)
log.write('make: '+petsc.make)
log.write('PETSc source directory: '+petsc.dir)
log.write('PETSc install directory: '+petsc.prefixdir)
log.write('PETSc version: '+petsc.lversion)
if not emptyarch:
  log.write('PETSc architecture: '+petsc.arch)
log.write('SLEPc source directory: '+slepc.dir)
if slepc.isinstall:
  log.write('SLEPc install directory: '+slepc.prefixdir)
log.write('SLEPc version: '+slepc.lversion)

# Clean previous configuration if needed
if archdirexisted:
  if not slepc.clean:
    try:
      f = open(os.path.join(confdir,'slepcvariables'),'r')
      searchlines = f.readlines()
      f.close()
      if any(pkg.packagename.upper() in ''.join(searchlines) for pkg in externalpackages) and not any(pkg.requested for pkg in externalpackages):
        log.Print('\nWARNING: forcing --with-clean=1 because previous configuration had external packages')
        slepc.clean = True
    except: pass
  if slepc.clean:
    log.Print('\nCleaning arch dir '+archdir+'...')
    try:
      for root, dirs, files in os.walk(archdir,topdown=False):
        for name in files:
          if name!='configure.log':
            os.remove(os.path.join(root,name))
    except:
      log.Exit('ERROR: Cannot remove existing files in '+archdir)
    for rdir in ['CMakeFiles','obj','externalpackages']:
      try:
        shutil.rmtree(os.path.join(archdir,rdir))
      except: pass

# Create other directories and configuration files
if not slepc.prefixdir:
  slepc.prefixdir = archdir
includedir = CreateDir(archdir,'include',log)
modulesdir = CreateDirTwo(confdir,'modules','slepc',log)
pkgconfdir = CreateDir(libdir,'pkgconfig',log)
slepcvars  = CreateFile(confdir,'slepcvariables',log)
slepcrules = CreateFile(confdir,'slepcrules',log)
slepcconf  = CreateFile(includedir,'slepcconf.h',log)
cmakeconf  = CreateFile(confdir,'SLEPcBuildInternal.cmake',log)
pkgconfig  = CreateFile(pkgconfdir,'SLEPc.pc',log)
if slepc.isinstall:
  modules  = CreateFile(modulesdir,slepc.lversion,log)
else:
  modules  = CreateFile(modulesdir,slepc.lversion+'-'+archname,log)
  reconfig = CreateFile(confdir,'reconfigure-'+archname+'.py',log)
  reconfigpath = os.path.join(confdir,'reconfigure-'+archname+'.py')

# Write initial part of file slepcvariables
slepcvars.write('SLEPC_CONFIGURE_OPTIONS = '+argdb.UsedArgs()+'\n')
slepcvars.write('SLEPC_DESTDIR = '+slepc.prefixdir+'\n')
if emptyarch:
  slepcvars.write('INSTALLED_PETSC = 1\n')
if slepc.datadir:
  slepcvars.write('DATAFILESPATH = '+slepc.datadir+'\n')

# Write initial part of file slepcconf.h
slepcconf.write('#if !defined(__SLEPCCONF_H)\n')
slepcconf.write('#define __SLEPCCONF_H\n\n')
AddDefine(slepcconf,'PETSC_DIR',petsc.dir)
AddDefine(slepcconf,'PETSC_ARCH',petsc.arch)
AddDefine(slepcconf,'DIR',slepc.dir)
AddDefine(slepcconf,'LIB_DIR',os.path.join(slepc.prefixdir,'lib'))
if slepc.isrepo:
  AddDefine(slepcconf,'VERSION_GIT',slepc.gitrev)
  AddDefine(slepcconf,'VERSION_DATE_GIT',slepc.gitdate)
  AddDefine(slepcconf,'VERSION_BRANCH_GIT',slepc.branch)

# Create global configuration file for the case of empty PETSC_ARCH
if emptyarch:
  globconf = CreateFile(os.path.join(slepc.dir,'lib','slepc','conf'),'slepcvariables',log)
  globconf.write('SLEPC_DIR = '+slepc.dir+'\n')
  globconf.write('PETSC_ARCH = '+archname+'\n')
  globconf.close()

# Check if PETSc is working
log.NewSection('Checking PETSc installation...')
if petsc.nversion > slepc.nversion:
  log.Println('\nWARNING: PETSc version '+petsc.version+' is newer than SLEPc version '+slepc.version)
if slepc.release=='1' and not petsc.release=='1':
  log.Exit('ERROR: a release version of SLEPc requires a release version of PETSc, not a development version')
if slepc.release=='0' and petsc.release=='1':
  log.Exit('ERROR: a development version of SLEPc cannot be built with a release version of PETSc')
if petsc.isinstall:
  if os.path.realpath(petsc.prefixdir) != os.path.realpath(petsc.dir):
    log.Println('\nWARNING: PETSC_DIR does not point to PETSc installation path')
petsc.Check()
if not petsc.havepackage:
  log.Exit('ERROR: Unable to link with PETSc')

# Single library installation
if petsc.singlelib:
  slepcvars.write('SHLIBS = libslepc\n')
  slepcvars.write('LIBNAME = '+os.path.join('${INSTALL_LIB_DIR}','libslepc.${AR_LIB_SUFFIX}')+'\n')
  for module in ['SYS','EPS','SVD','PEP','NEP','MFN','LME']:
    slepcvars.write('SLEPC_'+module+'_LIB = ${CC_LINKER_SLFLAG}${SLEPC_LIB_DIR} -L${SLEPC_LIB_DIR} -lslepc ${SLEPC_EXTERNAL_LIB} ${PETSC_SNES_LIB}\n')
  slepcvars.write('SLEPC_LIB = ${CC_LINKER_SLFLAG}${SLEPC_LIB_DIR} -L${SLEPC_LIB_DIR} -lslepc ${SLEPC_EXTERNAL_LIB} ${PETSC_SNES_LIB}\n')

# Check for external packages and for missing LAPACK functions
for pkg in checkpackages:
  pkg.Process(slepcconf,slepcvars,cmakeconf,petsc,archdir)

# Write Modules, pkg-config and CMake configuration files
log.NewSection('Writing various configuration files...')
log.write('Modules file in '+modulesdir)
if slepc.isinstall:
  WriteModulesFile(modules,slepc.lversion,slepc.prefixdir)
else:
  WriteModulesFile(modules,slepc.lversion,slepc.dir)
log.write('pkg-config file in '+pkgconfdir)
slflag = ''
if petsc.buildsharedlib: slflag = petsc.slflag
WritePkgconfigFile(pkgconfig,slepc.lversion,petsc.version,slepc.dir,slepc.isinstall,slepc.prefixdir,petsc.singlelib)
log.write('CMake configure file in '+confdir)
WriteCMakeConfigFile(cmakeconf)

# Write reconfigure file
if not slepc.isinstall:
  WriteReconfigScript(reconfig,slepc.dir,argdb.UsedArgs())
  try:
    os.chmod(reconfigpath,0o775)
  except OSError as e:
    log.Exit('ERROR: Unable to make reconfigure script executable:\n'+str(e))

# Finish with configuration files (except slepcvars)
slepcrules.close()
slepcconf.write('#endif\n')
slepcconf.close()
cmakeconf.close()
pkgconfig.close()
modules.close()
if not slepc.isinstall: reconfig.close()

# Download sowing if requested and make Fortran stubs if necessary
bfort = petsc.bfort
if sowing.downloadpackage:
  bfort = sowing.Install(archdir,petsc.make)

if slepc.isrepo and petsc.fortran:
  try:
    if not os.path.exists(bfort):
      bfort = os.path.join(archdir,'bin','bfort')
    if not os.path.exists(bfort):
      bfort = sowing.Install(archdir,petsc.make)
    log.NewSection('Generating Fortran stubs...')
    log.write('Using BFORT='+bfort)
    sys.path.insert(0, os.path.abspath(os.path.join('lib','slepc','bin','maint')))
    import generatefortranstubs
    generatefortranstubs.main(slepc.dir,bfort,os.getcwd(),0)
    generatefortranstubs.processf90interfaces(slepc.dir,0)
  except:
    log.Exit('ERROR: Try configuring with --download-sowing or use a git version of PETSc')

if bfort != petsc.bfort:
  slepcvars.write('BFORT = '+bfort+'\n')

# CMake stuff
cmakeok = False
if slepc.cmake:
  log.NewSection('Configuring CMake builds...')
  if sys.version_info < (2,5):
    log.Exit('ERROR: python version should be 2.5 or higher')
  elif petsc.isinstall:
    log.Exit('ERROR: CMake builds cannot be used with prefix-installed PETSc')
  elif not petsc.build_using_cmake:
    log.Exit('ERROR: CMake builds need a PETSc configured --with-cmake')
  else:
    import cmakegen
    try:
      cmakegen.main(slepc.dir,petsc.dir,petscprefixdir=petsc.prefixdir)
    except (OSError) as e:
      log.Exit('ERROR: Generating CMakeLists.txt failed:\n'+str(e))
    import cmakeboot
    try:
      cmakeok = cmakeboot.main(slepc.dir,petsc.dir,log=log)
    except (OSError) as e:
      log.Exit('ERROR: Booting CMake in PETSC_ARCH failed:\n'+str(e))
    except (ImportError, KeyError) as e:
      log.Exit('ERROR: Importing cmakeboot failed:\n'+str(e))
    except (AttributeError) as e:
      log.Println('\nxxx'+'='*74+'xxx')
      log.Println('WARNING: CMake builds are not available (initialization failed)')
      log.Println('You can ignore this warning (use default build), or try reconfiguring PETSc')
      log.Println('xxx'+'='*74+'xxx')
    # remove files created by PETSc's script
    for f in ['build.log','build.log.bkp','RDict.log']:
      try: os.remove(f)
      except OSError: pass
if cmakeok:
  slepcvars.write('SLEPC_BUILD_USING_CMAKE = 1\n')

# Finally we can close the slepcvariables file
slepcvars.close()

# Print summary
log.NewSection('')
log.Println('')
log.Println('='*80)
log.Println('SLEPc Configuration')
log.Println('='*80)
log.Println('\nSLEPc directory:\n '+slepc.dir)
if slepc.isrepo:
  log.Println('  It is a git repository on branch: '+slepc.branch)
if slepc.isinstall:
  log.Println('SLEPc prefix directory:\n '+slepc.prefixdir)
log.Println('PETSc directory:\n '+petsc.dir)
if petsc.isrepo:
  log.Println('  It is a git repository on branch: '+petsc.branch)
  if slepc.isrepo and petsc.branch!='maint' and slepc.branch!='maint':
    try:
      import dateutil.parser, datetime
      petscdate = dateutil.parser.parse(petsc.gitdate)
      slepcdate = dateutil.parser.parse(slepc.gitdate)
      if abs(petscdate-slepcdate)>datetime.timedelta(days=30):
        log.Println('xxx'+'='*74+'xxx')
        log.Println('WARNING: your PETSc and SLEPc repos may not be in sync (more than 30 days apart)')
        log.Println('xxx'+'='*74+'xxx')
    except ImportError: pass
if emptyarch and slepc.isinstall:
  log.Println('Prefix install with '+petsc.precision+' precision '+petsc.scalar+' numbers')
else:
  log.Println('Architecture "'+archname+'" with '+petsc.precision+' precision '+petsc.scalar+' numbers')
for pkg in checkpackages:
  pkg.ShowInfo()
log.write('\nFinishing Configure Run at '+time.ctime(time.time()))
log.write('='*80)
print()
print('xxx'+'='*74+'xxx')
if petsc.make_is_gnumake: buildtype = 'gnumake'
elif cmakeok: buildtype = 'cmake'
else: buildtype = 'legacy'
print(' Configure stage complete. Now build the SLEPc library with ('+buildtype+' build):')
if emptyarch:
  print('   make SLEPC_DIR=$PWD PETSC_DIR='+petsc.dir)
else:
  print('   make SLEPC_DIR=$PWD PETSC_DIR='+petsc.dir+' PETSC_ARCH='+archname)
print('xxx'+'='*74+'xxx')
print()
