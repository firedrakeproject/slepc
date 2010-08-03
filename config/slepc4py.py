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
import log
import petscconf
import urllib
import urlparse
import tarfile

def Install():
  '''
  Download and uncompress the slepc4py tarball
  '''
  log.Write('='*80)
  log.Println('Installing slepc4py...')

  # Test whether PETSc was installed with petsc4py
  petscdir = os.environ['PETSC_DIR']
  petscconf.Load(petscdir)
  archdir = os.sep.join([petscdir,petscconf.ARCH])
  if petscconf.ISINSTALL:
    petsc4pydir = os.sep.join([petscdir,'lib/petsc4py'])
  else:
    petsc4pydir = os.sep.join([archdir,'lib/petsc4py'])
  if not os.path.exists(petsc4pydir):
    sys.exit('ERROR: current PETSc configuration does not include petsc4py support' + petsc4pydir)

  # Create externalpackages directory
  externdir = 'externalpackages'
  if not os.path.exists(externdir):
    try:
      os.mkdir(externdir)
    except:
      sys.exit('ERROR: cannot create directory ' + externdir)

  # Download tarball
  packagename = 'slepc4py-1.0.0'
  url = 'http://slepc4py.googlecode.com/files/'+packagename+'.tar.gz'
  archiveZip = 'slepc4py.tgz'
  localFile = os.sep.join([externdir,archiveZip])
  log.Println('Downloading '+url+' to '+localFile)

  if os.path.exists(localFile):
    os.remove(localFile)
  try:
    urllib.urlretrieve(url, localFile)
  except Exception, e:
    name = 'slepc4py'
    filename   = os.path.basename(urlparse.urlparse(url)[2])
    failureMessage = '''\
Unable to download package %s from: %s
* If your network is disconnected - please reconnect and rerun config/configure.py
* Alternatively, you can download the above URL manually, to /yourselectedlocation/%s
  and use the configure option:
  --download-%s=/yourselectedlocation/%s
''' % (name, url, filename, name, filename)
    raise RuntimeError(failureMessage)

  # Uncompress tarball
  destDir = os.sep.join([externdir,packagename])
  log.Println('Uncompressing '+localFile+' to directory '+destDir)
  if os.path.exists(destDir):
    for root, dirs, files in os.walk(destDir, topdown=False):
      for name in files:
        os.remove(os.path.join(root, name))
      for name in dirs:
        os.rmdir(os.path.join(root, name))
  try:
    tar = tarfile.open(localFile, "r:gz")
    tar.extractall(path=externdir)
    tar.close()
  except RuntimeError, e:
    raise RuntimeError('Error uncompressing '+archiveZip+': '+str(e))

  os.remove(localFile)


def addMakeRule(slepcrules,installdir,prefixinstall,getslepc4py):
  '''
  Add a rule to the makefile in order to build slepc4py
  '''
  if getslepc4py:
    target = 'slepc4py'
    slepcrules.write(target+':\n')
    externdir = 'externalpackages'
    packagename = 'slepc4py-1.0.0'
    destDir = os.sep.join([externdir,packagename])
    cmd = '@cd '+destDir
    slepcrules.write('\t'+cmd+'; \\\n')
    cmd = 'python setup.py clean --all'
    slepcrules.write('\t'+cmd+'; \\\n')
    cmd = 'python setup.py install --install-lib='+os.path.join(installdir,'lib')
    slepcrules.write('\t'+cmd+'\n')
    cmd = '@echo "====================================="'
    slepcrules.write('\t'+cmd+'\n')
    cmd = '@echo "To use slepc4py, add '+os.path.join(installdir,'lib')+' to PYTHONPATH"'
    slepcrules.write('\t'+cmd+'\n')
    cmd = '@echo "====================================="'
    slepcrules.write('\t'+cmd+'\n')
  else:
    if prefixinstall:
      target = 'slepc4py'
      slepcrules.write(target+':\n')
      cmd = '@echo " "'
      slepcrules.write('\t'+cmd+'\n')

  target = 'slepc4py_noinstall'
  if getslepc4py and not prefixinstall:
    slepcrules.write(target+': slepc4py\n')
  else:
    slepcrules.write(target+':\n')
