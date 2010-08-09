#!/usr/bin/env python
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
import time

import petscversion
import petscconf
import log
import check
import arpack
import blzpack
import trlan  
import lapack
import primme
import slepc4py

if not hasattr(sys, 'version_info') or not sys.version_info[1] >= 2:
  print '**** You must have Python version 2.2 or higher to run config/configure.py ******'
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
  elif i.startswith('--download-slepc4py'):
    getslepc4py = not i.endswith('=0')
  elif i.startswith('--prefix='):
    prefixdir = i.split('=')[1]
  elif i.startswith('--h') or i.startswith('-h') or i.startswith('-?'):
    print 'SLEPc Configure Help'
    print '-'*80
    print '  --prefix=<dir>                   : Specifiy location to install SLEPc (eg. /usr/local)'
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
    print 'slepc4py:'
    print '  --download-slepc4py              : Download and install slepc4py in SLEPc directory'
    sys.exit(0)
  else:
    sys.exit('ERROR: Invalid argument ' + i +' use -h for help')

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
if petscversion.VERSION < '3.1':
  sys.exit('ERROR: This SLEPc version is not compatible with PETSc version '+petscversion.VERSION) 
if petscversion.PATCHLEVEL < '4' and petscversion.RELEASE == '1':
  sys.exit('ERROR: PETSc 3.1 patchlevel 4 required. Please upgrade to petsc-3.1-p4 or later') 

# Check some information about PETSc configuration
petscconf.Load(petscdir)
if not petscconf.PRECISION in ['double','single','matsingle']:
  sys.exit('ERROR: This SLEPc version does not work with '+petscconf.PRECISION+' precision')
if prefixdir and not petscconf.ISINSTALL:
  sys.exit('ERROR: SLEPc cannot be configured for non-source installation if PETSc is not configured in the same way.')

# Create architecture directory and configuration files
try:
  slepcvariables = open(os.sep.join([slepcdir,'conf','slepcvariables']),'w')
  slepcvariables.write('PETSC_DIR='+petscdir+'\n')
  slepcvariables.write('PETSC_ARCH='+petscconf.ARCH+'\n')
  slepcvariables.write('SLEPC_DIR='+slepcdir+'\n')
  slepcvariables.close() 
except:
  sys.exit('ERROR: cannot create default configuration file in ' + os.sep.join([slepcdir,'conf']))
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
try:
  slepcconf = open(os.sep.join([confdir,'slepcvariables']),'w')
  if not prefixdir:
    prefixdir = archdir
  slepcconf.write('SLEPC_INSTALL_DIR =' + prefixdir +'\n')
except:
  sys.exit('ERROR: cannot create configuration file in ' + confdir)
try:
  slepcrules = open(os.sep.join([confdir,'slepcrules']),'w')
except:
  sys.exit('ERROR: cannot create rules file in ' + confdir)
if prefixinstall and os.path.isfile(os.sep.join([prefixdir,'include/slepc.h'])):
  sys.exit('ERROR: prefix directory ' + prefixdir + ' contains files from a previous installation')

# Open log file
log.Open(os.sep.join([confdir,'configure.log']))
log.Write('='*80)
log.Write('Starting Configure Run at '+time.ctime(time.time()))
log.Write('Configure Options: '+str.join(' ',sys.argv))
log.Write('Working directory: '+os.getcwd())
log.Write('Python version:\n' + sys.version)
log.Write('make: ' + petscconf.MAKE)
log.Write('PETSc source directory: ' + petscdir)
log.Write('PETSc install directory: ' + petscconf.INSTALL_DIR)
log.Write('PETSc version: ' + petscversion.VERSION)
log.Write('PETSc architecture: ' + petscconf.ARCH)
log.Write('SLEPc source directory: ' + slepcdir)
log.Write('SLEPc install directory: ' + prefixdir)
log.Write('='*80)

# Check if PETSc is working
log.Println('Checking PETSc installation...')
if petscversion.VERSION > '3.1':
  log.Println('WARNING: PETSc version '+petscversion.VERSION+' is newer than SLEPc version')
if petscversion.RELEASE != '1':
  log.Println('WARNING: using PETSc development version')
if petscconf.ISINSTALL:
  if os.path.realpath(petscconf.INSTALL_DIR) != os.path.realpath(petscdir):
    log.Println('WARNING: PETSC_DIR does not point to PETSc installation path')
if not check.Link([],[],[]):
  log.Exit('ERROR: Unable to link with PETSc')

# Check for external packages
if havearpack:
  arpacklibs = arpack.Check(slepcconf,arpackdir,arpacklibs)
if haveblzpack:
  blzpacklibs = blzpack.Check(slepcconf,blzpackdir,blzpacklibs)
if havetrlan:
  trlanlibs = trlan.Check(slepcconf,trlandir,trlanlibs)
if haveprimme:
  primmelibs = primme.Check(slepcconf,primmedir,primmelibs)

# Check for missing LAPACK functions
missing = lapack.Check(slepcconf)

# Download and install slepc4py
if getslepc4py:
  slepc4py.Install()
slepc4py.addMakeRule(slepcrules,prefixdir,prefixinstall,getslepc4py)

slepcconf.close()
slepcrules.close()

log.Println('')
log.Println('='*80)
log.Println('SLEPc Configuration')
log.Println('='*80)
log.Println('')
log.Println('SLEPc source directory:')
log.Println(' '+slepcdir)
log.Println('SLEPc install directory:')
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
if missing:
  log.Println('LAPACK missing functions:')
  log.Print('  ')
  for i in missing: log.Print(i)
  log.Println('')
  log.Println('')
  log.Println('WARNING: Some SLEPc functionality will not be available')
  log.Println('PLEASE reconfigure and recompile PETSc with a full LAPACK implementation')
if petscconf.ISINSTALL:  
  log.Println('')
  log.Println('  **')
  log.Println('  ** Before running "make" your PETSC_ARCH must be specified with:')
  log.Println('  **  ** setenv PETSC_ARCH '+petscconf.ARCH+' (csh/tcsh)')
  log.Println('  **  ** PETSC_ARCH='+petscconf.ARCH+' ; export PETSC_ARCH (sh/bash)')
  log.Println('  **')
print
