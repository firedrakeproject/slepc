#!/usr/bin/env python

import os
import sys

import petscconf
import log
import check
import arpack
import blzpack
import planso
import trlan  
import lapack

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
haveplanso = 0
plansodir = ''
plansolibs = []
havetrlan = 0
trlandir = ''
trlanlibs = []

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
  elif i.startswith('--with-planso-dir='):
    plansodir = i.split('=')[1]
    haveplanso = 1
  elif i.startswith('--with-planso-flags='):
    plansolibs = i.split('=')[1].split(',')
    haveplanso = 1
  elif i.startswith('--with-planso'):
    haveplanso = not i.endswith('=0')
  elif i.startswith('--with-trlan-dir='):
    trlandir = i.split('=')[1]
    havetrlan = 1
  elif i.startswith('--with-trlan-flags='):
    trlanlibs = i.split('=')[1].split(',')
    havetrlan = 1
  elif i.startswith('--with-trlan'):
    havetrlan = not i.endswith('=0')
  elif i.startswith('--h') or i.startswith('-h') or i.startswith('-?'):
    print 'HELP!'
    sys.exit(0)
  else:
    sys.exit('ERROR: Invalid argument ' + i +' use -h for help')

# Check if enviroment is ok
# and get some information about PETSc configuration
print 'Checking environment...'

if 'SLEPC_DIR' not in os.environ:
  sys.exit('ERROR: SLEPC_DIR enviroment variable is not set')
slepcdir = os.environ['SLEPC_DIR']
if not os.path.exists(slepcdir) or not os.path.exists(os.sep.join([slepcdir,'bmake'])):
  sys.exit('ERROR: SLEPC_DIR enviroment variable is not valid')
os.chdir(slepcdir);

if 'PETSC_DIR' not in os.environ:
  sys.exit('ERROR: PETSC_DIR enviroment variable is not set')
petscdir = os.environ['PETSC_DIR']
if not os.path.exists(petscdir) or not os.path.exists(os.sep.join([petscdir,'bmake'])):
  sys.exit('ERROR: PETSC_DIR enviroment variable is not valid')

petscconf.Load(petscdir)

log.Open('configure_log_' + petscconf.ARCH)

# Check if PETSc is working
log.Println('Checking PETSc library...')
if not check.Link([],[],[]):
  log.Exit('ERROR: PETSc is not installed correctly')

# Create architecture directory
archdir = os.sep.join([slepcdir,'bmake',petscconf.ARCH])
if not os.path.exists(archdir):
  try:
    os.mkdir(archdir)
  except:
    log.Exit('ERROR: cannot create architecture directory ' + archdir)

slepcconf = open(os.sep.join([archdir,'slepcconf']),'w')

# Check for missing LAPACK functions
log.Println('Checking LAPACK library...')
missing = lapack.Check(slepcconf)

# Check for external packages
if havearpack:
  arpacklibs = arpack.Check(slepcconf,arpackdir,arpacklibs)
if haveblzpack:
  blzpacklibs = blzpack.Check(slepcconf,blzpackdir,blzpacklibs)
if haveplanso:
  plansolibs = planso.Check(slepcconf,plansodir,plansolibs)
if havetrlan:
  trlanlibs = trlan.Check(slepcconf,trlandir,trlanlibs)

slepcconf.close()

log.Println('')
log.Println('='*80)
log.Println('SLEPc Configuration')
log.Println('='*80)
log.Println('')
log.Println('SLEPc directory:')
log.Println(' '+slepcdir)
log.Println('PETSc directory:')
log.Println(' '+petscdir)
log.Println('Architecture "'+petscconf.ARCH+'" with '+petscconf.PRECISION+' precision '+petscconf.SCALAR+' numbers')
if petscconf.MPIUNI:
  log.Println('  Uniprocessor version without MPI')
if havearpack:
  log.Println('ARPACK library flags:')
  log.Println(' '+str.join(' ',arpacklibs))
if haveblzpack:
  log.Println('BLZPACK library flags:')
  log.Println(' '+str.join(' ',blzpacklibs))
if haveplanso:
  log.Println('PLANSO library flags:')
  log.Println(' '+str.join(' ',plansolibs))
if missing:
  log.Println('LAPACK mising functions:')
  log.Print('  ')
  for i in missing: log.Print(i)
  log.Println('')
  log.Println('')
  log.Println('WARNING: Some SLEPc functionality will not be avaliable')
  log.Println('PLEASE reconfigure and recompile PETSc with a full LAPACK implementation')
print
