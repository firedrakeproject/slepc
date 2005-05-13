#!/usr/bin/env python

import os
import sys

from check   import *
from arpack  import *
from blzpack import *
from planso  import *
from trlan   import *
from lapack  import *

if not hasattr(sys, 'version_info') or not sys.version_info[1] >= 2:
  print '**** You must have Python version 2.2 or higher to run config/configure.py ******'
  print '*           Python is easy to install for end users or sys-admin.               *'
  print '*                   http://www.python.org/download/                             *'
  print '*                                                                               *'
  print '*            You CANNOT configure PETSc without Python                          *'
  print '*    http://www.mcs.anl.gov/petsc/petsc-as/documentation/installation.html      *'
  print '*********************************************************************************'
  sys.exit(4)

# Check if enviroment is ok
# and get some information about PETSc configuration

if 'SLEPC_DIR' not in os.environ:
  sys.exit('ERROR: SLEPC_DIR enviroment variable is not set')
slepcdir = os.environ['SLEPC_DIR']
if not os.path.exists(slepcdir) or not os.path.exists(os.sep.join([slepcdir,'bmake'])):
  sys.exit('ERROR: SLEPC_DIR enviroment variable is not valid')
  
if os.getcwd() != slepcdir:
  sys.exit('ERROR: configure.py must be launched from SLEPc main directory')

if 'PETSC_DIR' not in os.environ:
  sys.exit('ERROR: PETSC_DIR enviroment variable is not set')
petscdir = os.environ['PETSC_DIR']
if not os.path.exists(petscdir) or not os.path.exists(os.sep.join([petscdir,'bmake'])):
  sys.exit('ERROR: PETSC_DIR enviroment variable is not valid')

if 'PETSC_ARCH' in os.environ:
  petscarch = os.environ['PETSC_ARCH']
else:
  try:
    f = open(os.sep.join([petscdir,'bmake','petscconf']))
    petscarch = ''
    for l in f.readlines():
      if l.startswith('PETSC_ARCH='):
	petscarch = l.split('=')[1].rstrip()
	f.close()
	break
    f.close()
  except:
    sys.exit('ERROR: PETSc must be configured first')
  if not petscarch:
    sys.exit('ERROR: please set enviroment variable PETSC_ARCH')

petscscalar = ''
petscprecision = ''
petscmpiuni = 0
try:
  f = open(os.sep.join([petscdir,'bmake',petscarch,'petscconf']))
  for l in f.readlines():
    l = l.rstrip()
    if l.startswith('PETSC_SCALAR'):
      petscscalar = l.split('=')[1].lstrip()
    elif l.startswith('PETSC_PRECISION'):
      petscprecision = l.split('=')[1].lstrip()
    elif l.startswith('MPI_INCLUDE') and l.endswith('mpiuni'):
      petscmpiuni = 1
    elif l.startswith('MAKE'):
      MAKE = l.split('=')[1].lstrip()
  f.close()
except:
  sys.exit('ERROR: PETSc is not configured for architecture ' + petscarch)

# Check if PETSc is working
if not checkLink([],[],[]):
  sys.exit('ERROR: PETSc is not installed correctly')

# Create architecture directory
archdir = os.sep.join([slepcdir,'bmake',petscarch])
if not os.path.exists(archdir):
  try:
    os.mkdir(archdir)
  except:
    sys.exit('ERROR: cannot create architecture directory ' + archdir)
slepcconf = open(os.sep.join([archdir,'slepcconf']),'w')

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
  elif i.startswith('--with-arpack-lib='):
    arpacklibs = i.split('=')[1].split(',')
    havearpack = 1
  elif i.startswith('--with-arpack'):
    havearpack = not i.endswith('=0')
  elif i.startswith('--with-blzpack-dir='):
    blzpackdir = i.split('=')[1]
    haveblzpack = 1
  elif i.startswith('--with-blzpack-lib='):
    blzpacklibs = i.split('=')[1].split(',')
    haveblzpack = 1
  elif i.startswith('--with-blzpack'):
    haveblzpack = not i.endswith('=0')
  elif i.startswith('--with-planso-dir='):
    plansodir = i.split('=')[1]
    haveplanso = 1
  elif i.startswith('--with-planso-lib='):
    plansolibs = i.split('=')[1].split(',')
    haveplanso = 1
  elif i.startswith('--with-planso'):
    haveplanso = not i.endswith('=0')
  elif i.startswith('--with-trlan-dir='):
    trlandir = i.split('=')[1]
    havetrlan = 1
  elif i.startswith('--with-trlan-lib='):
    trlanlibs = i.split('=')[1].split(',')
    havetrlan = 1
  elif i.startswith('--with-trlan'):
    havetrlan = not i.endswith('=0')
  else:
    sys.exit('ERROR: Invalid argument ' + i)

# Check for external packages
if havearpack:
  arpacklibs = checkArpack(slepcconf,arpackdir,arpacklibs,petscscalar,petscprecision,petscmpiuni)
if haveblzpack:
  blzpacklibs = checkBlzpack(slepcconf,blzpackdir,blzpacklibs,petscscalar,petscprecision,petscmpiuni)
if haveplanso:
  plansolibs = checkPlanso(slepcconf,plansodir,plansolibs,petscscalar,petscprecision,petscmpiuni)
if havetrlan:
  trlanlibs = checkTrlan(slepcconf,trlandir,trlanlibs,petscscalar,petscprecision,petscmpiuni)

# Check for missing LAPACK functions
missing = checkLapack(slepcconf,petscscalar,petscprecision)

slepcconf.close()

print
print '='*80
print 'SLEPc Configuration'
print '='*80
print
print 'SLEPc directory:'
print ' ',slepcdir
print 'PETSc directory:'
print ' ',petscdir
print 'Architecture "%s" with %s precision %s numbers' % (petscarch,petscprecision,petscscalar)
if petscmpiuni:
  print '  Uniprocessor version'
if havearpack:
  print 'ARPACK library flags:'
  print ' ',str.join(' ',arpacklibs)
if haveblzpack:
  print 'BLZPACK library flags:'
  print ' ',str.join(' ',blzpacklibs)
if haveplanso:
  print 'PLANSO library flags:'
  print ' ',str.join(' ',plansolibs)
if missing:
  print 'LAPACK mising functions:'
  print '  ',
  for i in missing: print i,
  print
  print
  print 'WARNING: some SLEPc functionality will not be avaliable'
  print 'PLEASE reconfigure and recompile PETSc with a full LAPACK implementation'
print