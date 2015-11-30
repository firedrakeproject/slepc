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

import os, sys, commands, shutil
import urllib, urlparse
import log, package

class Primme(package.Package):

  def __init__(self,argdb,log):
    self.packagename     = 'primme'
    self.installable     = True
    self.downloadable    = True
    self.downloadpackage = 0
    self.packagedir      = ''
    self.packagelibs     = []
    self.log             = log
    self.ProcessArgs(argdb)

  def Check(self,conf,vars,cmake,petsc):

    if petsc.precision != 'double':
      self.log.Exit('ERROR: PRIMME is supported only in double precision.')

    if petsc.ind64:
      self.log.Exit('ERROR: Cannot use external packages with 64-bit indices.')

    functions_base = ['primme_set_method','primme_Free','primme_initialize']
    if self.packagedir:
      dirs = [self.packagedir]
    else:
      dirs = self.GenerateGuesses('Primme')

    libs = self.packagelibs
    if not libs:
      libs = ['-lprimme']
    if petsc.scalar == 'real':
      functions = functions_base + ['dprimme']
    else:
      functions = functions_base + ['zprimme']

    for d in dirs:
      if d:
        if 'rpath' in petsc.slflag:
          l = [petsc.slflag + d] + ['-L' + d] + libs
        else:
          l = ['-L' + d] + libs
        f = ['-I' + os.path.join(d,'PRIMMESRC','COMMONSRC')]
      else:
        l =  libs
        f = []
      if self.Link(functions,[],l+f):
        conf.write('#ifndef SLEPC_HAVE_PRIMME\n#define SLEPC_HAVE_PRIMME 1\n#endif\n\n')
        vars.write('PRIMME_LIB = ' + ' '.join(l) + '\n')
        vars.write('PRIMME_FLAGS = ' + ' '.join(f) + '\n')
        cmake.write('set (SLEPC_HAVE_PRIMME YES)\n')
        cmake.write('find_library (PRIMME_LIB primme HINTS '+ d +')\n')
        cmake.write('find_path (PRIMME_INCLUDE primme.h ' + d + '/PRIMMESRC/COMMONSRC)\n')
        self.havepackage = True
        self.packageflags = l+f
        return

    self.log.Println('\nERROR: Unable to link with PRIMME library')
    self.log.Println('ERROR: In directories '+' '.join(dirs))
    self.log.Println('ERROR: With flags '+' '.join(libs))
    self.log.Exit('')


  def Install(self,conf,vars,cmake,petsc,archdir):
    '''
    Download and uncompress the PRIMME tarball
    '''
    if petsc.precision != 'double':
      self.log.Exit('ERROR: PRIMME is supported only in double precision.')

    if petsc.ind64:
      self.log.Exit('ERROR: Cannot use external packages with 64-bit indices.')

    packagename = 'PRIMME'
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
        url = 'https://github.com/primme/primme/tarball/release-1.2.2'
      archiveZip = 'primme-1.2.2.tar.gz'
      localFile = os.path.join(externdir,archiveZip)
      self.log.write('Downloading '+url+' to '+localFile)

      if os.path.exists(localFile):
        os.remove(localFile)
      try:
        urllib.urlretrieve(url, localFile)
      except Exception, e:
        name = 'primme'
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

      # Rename directory
      for filename in os.listdir(externdir):
        if filename.startswith('primme-'):
          os.rename(os.path.join(externdir,filename),builddir)

    # Configure
    g = open(os.path.join(builddir,'Make_flags'),'w')
    g.write('LIBRARY     = libprimme.a\n')
    g.write('DLIBRARY    = libdprimme.a\n')
    g.write('ZLIBRARY    = libzprimme.a\n')
    g.write('CC          = '+petsc.cc+'\n')
    g.write('DEFINES     = -DF77UNDERSCORE\n')
    g.write('CFLAGS      = '+petsc.cc_flags.replace('-Wall','').replace('-Wshadow','')+'\n')
    g.write('RANLIB      = '+petsc.ranlib+'\n')
    g.close()

    # Build package
    result,output = commands.getstatusoutput('cd '+builddir+'&&'+petsc.make+' clean &&'+petsc.make)
    self.log.write(output)

    # Move files
    incDir = os.path.join(archdir,'include')
    libDir = os.path.join(archdir,'lib')
    os.rename(os.path.join(builddir,'libprimme.'+petsc.ar_lib_suffix),os.path.join(libDir,'libprimme.'+petsc.ar_lib_suffix))
    for name in ['primme.h','primme_f77.h','Complexz.h']:
      shutil.copyfile(os.path.join(builddir,'PRIMMESRC','COMMONSRC',name),os.path.join(incDir,name))

    if 'rpath' in petsc.slflag:
      l = petsc.slflag + libDir + ' -L' + libDir + ' -lprimme'
    else:
      l = '-L' + libDir + ' -lprimme'
    f = '-I' + incDir

    # Check build
    functions_base = ['primme_set_method','primme_Free','primme_initialize']
    if petsc.scalar == 'real':
      functions = functions_base + ['dprimme']
    else:
      functions = functions_base + ['zprimme']
    if not self.Link(functions,[],[l]+[f]):
      self.log.Exit('\nERROR: Unable to link with downloaded PRIMME')

    # Write configuration files
    conf.write('#ifndef SLEPC_HAVE_PRIMME\n#define SLEPC_HAVE_PRIMME 1\n#endif\n\n')
    vars.write('PRIMME_LIB = ' + l + '\n')
    cmake.write('set (SLEPC_HAVE_PRIMME YES)\n')
    cmake.write('find_library (PRIMME_LIB primme HINTS '+ libDir +')\n')
    cmake.write('find_path (PRIMME_INCLUDE primme.h ' + incDir + ')\n')

    self.havepackage = True
    self.packageflags = [l] + [f]

