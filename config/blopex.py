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
import log
import petscconf
import urllib
import urlparse
import commands

def Install(conf,vars,cmake,tmpdir,url,archdir):
  '''
  Download and uncompress the BLOPEX tarball
  '''
  log.write('='*80)
  log.Println('Installing BLOPEX...')

  if petscconf.PRECISION != 'double':
    log.Exit('ERROR: BLOPEX is supported only in double precision.')

  if petscconf.IND64:
    log.Exit('ERROR: cannot use external packages with 64-bit indices.')

  packagename = 'blopex-1.1.2'
  externdir   = archdir+'/externalpackages'
  builddir    = os.sep.join([externdir,packagename])

  # Create externalpackages directory
  if not os.path.exists(externdir):
    try:
      os.mkdir(externdir)
    except:
      sys.exit('ERROR: cannot create directory ' + externdir)

  # Check if source is already available
  if os.path.exists(builddir):
    log.Println('Using '+builddir)
  else:

    # Download tarball
    if url=='':
      url = 'http://slepc.upv.es/download/external/'+packagename+'.tar.gz'
    archiveZip = 'blopex.tar.gz'
    localFile = os.sep.join([externdir,archiveZip])
    log.Println('Downloading '+url+' to '+localFile)
  
    if os.path.exists(localFile):
      os.remove(localFile)
    try:
      urllib.urlretrieve(url, localFile)
    except Exception, e:
      name = 'blopex'
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
    log.Println('Uncompressing '+localFile+' to directory '+builddir)
    if os.path.exists(builddir):
      for root, dirs, files in os.walk(builddir, topdown=False):
        for name in files:
          os.remove(os.path.join(root, name))
        for name in dirs:
          os.rmdir(os.path.join(root, name))
    try:
      if sys.version_info >= (2,5):
        import tarfile
        tar = tarfile.open(localFile, "r:gz")
        tar.extractall(path=externdir)
        tar.close()
        os.remove(localFile)
      else:
        result,output = commands.getstatusoutput('cd '+externdir+'; gunzip '+archiveZip+'; tar -xf '+archiveZip.split('.gz')[0])
        os.remove(localFile.split('.gz')[0])
    except RuntimeError, e:
      raise RuntimeError('Error uncompressing '+archiveZip+': '+str(e))

  # Configure
  g = open(os.path.join(builddir,'Makefile.inc'),'w')
  g.write('CC          = '+petscconf.CC+'\n')
  if petscconf.IND64: blopexint = ' -DBlopexInt="long long" '
  else: blopexint = ''
  g.write('CFLAGS      = '+petscconf.CC_FLAGS.replace('-Wall','').replace('-Wshadow','')+blopexint+'\n')
  g.write('AR          = '+petscconf.AR+' '+petscconf.AR_FLAGS+'\n')
  g.write('AR_LIB_SUFFIX = '+petscconf.AR_LIB_SUFFIX+'\n')
  g.write('RANLIB      = '+petscconf.RANLIB+'\n')
  g.write('TARGET_ARCH = \n')
  g.close()

  # Build package
  result,output = commands.getstatusoutput('cd '+builddir+'&&'+petscconf.MAKE+' clean &&'+petscconf.MAKE)
  log.write(output)

  # Move files
  incDir = os.sep.join([archdir,'include'])
  libDir = os.sep.join([archdir,'lib'])
  os.rename(os.path.join(builddir,'lib/libBLOPEX.'+petscconf.AR_LIB_SUFFIX),os.path.join(libDir,'libBLOPEX.'+petscconf.AR_LIB_SUFFIX))
  for root, dirs, files in os.walk(os.path.join(builddir,'include')):
    for name in files:
      os.rename(os.path.join(builddir,'include/'+name),os.path.join(incDir,name))

  if 'rpath' in petscconf.SLFLAG:
    l = petscconf.SLFLAG + libDir + ' -L' + libDir + ' -lBLOPEX'
  else:
    l = '-L' + libDir + ' -lBLOPEX'

  # Write configuration files
  conf.write('#ifndef SLEPC_HAVE_BLOPEX\n#define SLEPC_HAVE_BLOPEX 1\n#endif\n\n')
  vars.write('BLOPEX_LIB = ' + l + '\n')
  cmake.write('set (SLEPC_HAVE_BLOPEX YES)\n')
  cmake.write('find_library (BLOPEX_LIB BLOPEX HINTS '+ libDir +')\n')

  return [l] + ['-I' + incDir]

