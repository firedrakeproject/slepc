#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain
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

import os, sys, commands
import urllib, urlparse
import log, package

class Blopex(package.Package):

  def __init__(self,argdb,log):
    self.packagename     = 'blopex'
    self.downloadable    = True
    self.downloadpackage = 0
    self.packageurl      = ''
    self.log             = log
    self.ProcessArgs(argdb)

  def Install(self,conf,vars,cmake,petsc,archdir):
    '''
    Download and uncompress the BLOPEX tarball
    '''
    if petsc.precision != 'double':
      self.log.Exit('ERROR: BLOPEX is supported only in double precision.')

    if petsc.ind64:
      self.log.Exit('ERROR: Cannot use external packages with 64-bit indices.')

    packagename = 'blopex-1.1.2'
    externdir   = os.path.join(archdir,'externalpackages')
    builddir    = os.path.join(externdir,packagename)

    # Create externalpackages directory
    if not os.path.exists(externdir):
      try:
        os.mkdir(externdir)
      except:
        self.log.Exit('ERROR: Cannot create directory ' + externdir)

    # Check if source is already available
    if os.path.exists(builddir):
      self.log.write('Using '+builddir)
    else:

      # Download tarball
      url = self.packageurl
      if url=='':
        url = 'http://slepc.upv.es/download/external/'+packagename+'.tar.gz'
      archiveZip = 'blopex.tar.gz'
      localFile = os.path.join(externdir,archiveZip)
      self.log.write('Downloading '+url+' to '+localFile)

      if os.path.exists(localFile):
        os.remove(localFile)
      try:
        urllib.urlretrieve(url, localFile)
      except Exception, e:
        name = 'blopex'
        filename = os.path.basename(urlparse.urlparse(url)[2])
        failureMessage = '''\
Unable to download package %s from: %s
* If your network is disconnected - please reconnect and rerun ./configure
* Alternatively, you can download the above URL manually, to /yourselectedlocation/%s
  and use the configure option:
  --download-%s=/yourselectedlocation/%s
''' % (name, url, filename, name, filename)
        self.log.Exit(failureMessage)

      # Uncompress tarball
      self.log.write('Uncompressing '+localFile+' to directory '+builddir)
      if os.path.exists(builddir):
        for root, dirs, files in os.walk(builddir, topdown=False):
          for name in files:
            os.remove(os.path.join(root,name))
          for name in dirs:
            os.rmdir(os.path.join(root,name))
      try:
        if sys.version_info >= (2,5):
          import tarfile
          tar = tarfile.open(localFile, 'r:gz')
          tar.extractall(path=externdir)
          tar.close()
          os.remove(localFile)
        else:
          result,output = commands.getstatusoutput('cd '+externdir+'; gunzip '+archiveZip+'; tar -xf '+archiveZip.split('.gz')[0])
          os.remove(localFile.split('.gz')[0])
      except RuntimeError, e:
        self.log.Exit('Error uncompressing '+archiveZip+': '+str(e))

    # Configure
    g = open(os.path.join(builddir,'Makefile.inc'),'w')
    g.write('CC          = '+petsc.cc+'\n')
    if petsc.ind64: blopexint = ' -DBlopexInt="long long" '
    else: blopexint = ''
    g.write('CFLAGS      = '+petsc.cc_flags.replace('-Wall','').replace('-Wshadow','')+blopexint+'\n')
    g.write('AR          = '+petsc.ar+' '+petsc.ar_flags+'\n')
    g.write('AR_LIB_SUFFIX = '+petsc.ar_lib_suffix+'\n')
    g.write('RANLIB      = '+petsc.ranlib+'\n')
    g.write('TARGET_ARCH = \n')
    g.close()

    # Build package
    result,output = commands.getstatusoutput('cd '+builddir+'&&'+petsc.make+' clean &&'+petsc.make)
    self.log.write(output)

    # Move files
    incDir = os.path.join(archdir,'include')
    libDir = os.path.join(archdir,'lib')
    os.rename(os.path.join(builddir,'lib','libBLOPEX.'+petsc.ar_lib_suffix),os.path.join(libDir,'libBLOPEX.'+petsc.ar_lib_suffix))
    for root, dirs, files in os.walk(os.path.join(builddir,'include')):
      for name in files:
        os.rename(os.path.join(builddir,'include',name),os.path.join(incDir,name))

    if 'rpath' in petsc.slflag:
      l = petsc.slflag + libDir + ' -L' + libDir + ' -lBLOPEX'
    else:
      l = '-L' + libDir + ' -lBLOPEX'

    # Write configuration files
    conf.write('#ifndef SLEPC_HAVE_BLOPEX\n#define SLEPC_HAVE_BLOPEX 1\n#endif\n\n')
    vars.write('BLOPEX_LIB = ' + l + '\n')
    cmake.write('set (SLEPC_HAVE_BLOPEX YES)\n')
    cmake.write('find_library (BLOPEX_LIB BLOPEX HINTS '+ libDir +')\n')

    self.havepackage = True
    self.packageflags = [l] + ['-I' + incDir]

