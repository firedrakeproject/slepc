#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os, shutil, log, package

class Primme(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename    = 'primme'
    self.installable    = True
    self.downloadable   = True
    self.buildflags     = ''
    self.version        = '3.0'
    self.url            = 'https://github.com/primme/primme/tarball/release-'+self.version
    self.archive        = 'primme-'+self.version+'.tar.gz'
    self.dirname        = 'PRIMME'
    self.supportssingle = True
    self.supports64bint = True
    self.ProcessArgs(argdb)

  def ProcessArgs(self,argdb):
    string,found = argdb.PopString('download-'+self.packagename+'-cflags')
    if found:
      self.buildflags = string
    package.Package.ProcessArgs(self,argdb)

  def ShowHelp(self):
    package.Package.ShowHelp(self)
    print(('  --download-'+self.packagename+'-cflags=<flags>').ljust(package.Package.wd)+': Extra flags to compile '+self.packagename.upper())
 
  def Check(self,conf,vars,petsc):
    functions_base = ['primme_set_method','primme_free','primme_initialize']
    if self.packagedir:
      dirs = [os.path.join(self.packagedir,'lib')]
    else:
      dirs = self.GenerateGuesses('Primme')

    libs = self.packagelibs
    if not libs:
      libs = ['-lprimme']
    if petsc.scalar == 'real':
      if petsc.precision == 'single':
        functions = functions_base + ['sprimme']
      else:
        functions = functions_base + ['dprimme']
    else:
      if petsc.precision == 'single':
        functions = functions_base + ['cprimme']
      else:
        functions = functions_base + ['zprimme']

    for d in dirs:
      if d:
        if petsc.buildsharedlib:
          l = [petsc.slflag + d] + ['-L' + d] + libs
        else:
          l = ['-L' + d] + libs
        f = ['-I' + os.path.join(os.path.dirname(d),'include')]
      else:
        l =  libs
        f = []
      if self.Link(functions,[],l+f):
        conf.write('#define SLEPC_HAVE_PRIMME 1\n')
        vars.write('PRIMME_LIB = ' + ' '.join(l) + '\n')
        vars.write('PRIMME_FLAGS = ' + ' '.join(f) + '\n')
        self.havepackage = True
        self.packageflags = l+f
        self.location = os.path.dirname(d)
        return

    self.log.Println('\nERROR: Unable to link with PRIMME library')
    self.log.Println('ERROR: In directories '+' '.join(dirs))
    self.log.Println('ERROR: With flags '+' '.join(libs))
    self.log.Println('NOTE: make sure PRIMME version is 2.0 at least')
    self.log.Exit('')


  def Install(self,conf,vars,petsc,archdir):
    externdir = os.path.join(archdir,'externalpackages')
    builddir  = os.path.join(externdir,self.dirname)
    self.Download(externdir,builddir,'primme-')

    # Configure
    g = open(os.path.join(builddir,'mymake_flags'),'w')
    g.write('export LIBRARY     = libprimme.'+petsc.ar_lib_suffix+'\n')
    g.write('export SOLIBRARY   = libprimme.'+petsc.sl_suffix+'\n')
    g.write('export SONAMELIBRARY = libprimme.'+petsc.sl_suffix+self.version+'\n')
    g.write('export CC          = '+petsc.cc+'\n')
    if hasattr(petsc,'fc'):
      g.write('export F77         = '+petsc.fc+'\n')
    g.write('export DEFINES     = ')
    if petsc.blaslapackmangling == 'underscore':
      g.write('-DF77UNDERSCORE ')
    if petsc.blaslapackint64:
      g.write('-DPRIMME_BLASINT_SIZE=64')
    g.write('\n')
    g.write('export INCLUDE     = \n')
    g.write('export CFLAGS      = '+petsc.cc_flags.replace('-Wall','').replace('-Wshadow','').replace('-fvisibility=hidden','')+' '+self.buildflags+'\n')
    g.write('export RANLIB      = '+petsc.ranlib+'\n')
    g.write('export PREFIX      = '+archdir+'\n')
    g.write('include makefile\n')
    g.close()

    # Build package
    target = ' install' if petsc.buildsharedlib else ' lib'
    mymake = petsc.make + ' -f mymake_flags '
    result,output = self.RunCommand('cd '+builddir+'&&'+mymake+' clean && '+mymake+target)
    self.log.write(output)
    if result:
      self.log.Exit('ERROR: installation of PRIMME failed.')

    # Move files
    incDir = os.path.join(archdir,'include')
    libDir = os.path.join(archdir,'lib')
    if not petsc.buildsharedlib:
      os.rename(os.path.join(builddir,'lib','libprimme.'+petsc.ar_lib_suffix),os.path.join(libDir,'libprimme.'+petsc.ar_lib_suffix))
      for root, dirs, files in os.walk(os.path.join(builddir,'include')):
        for name in files:
          shutil.copyfile(os.path.join(builddir,'include',name),os.path.join(incDir,name))

    if petsc.buildsharedlib:
      l = petsc.slflag + libDir + ' -L' + libDir + ' -lprimme'
    else:
      l = '-L' + libDir + ' -lprimme'
    f = '-I' + incDir

    # Check build
    functions_base = ['primme_set_method','primme_free','primme_initialize']
    if petsc.scalar == 'real':
      if petsc.precision == 'single':
        functions = functions_base + ['sprimme']
      else:
        functions = functions_base + ['dprimme']
    else:
      if petsc.precision == 'single':
        functions = functions_base + ['cprimme']
      else:
        functions = functions_base + ['zprimme']
    if not self.Link(functions,[],[l]+[f]):
      self.log.Exit('\nERROR: Unable to link with downloaded PRIMME')

    # Write configuration files
    conf.write('#define SLEPC_HAVE_PRIMME 1\n')
    vars.write('PRIMME_LIB = ' + l + '\n')

    self.location = archdir
    self.havepackage = True
    self.packageflags = [l] + [f]


  def LoadVersion(self,conf):
    try:
      f = open(os.path.join(self.location,'include','primme.h'))
      for l in f.readlines():
        l = l.split()
        if len(l) == 3:
          if l[1] == 'PRIMME_VERSION_MAJOR':
            major = l[2]
          elif l[1] == 'PRIMME_VERSION_MINOR':
            minor = l[2]
      f.close()
      self.iversion = major + '.' + minor
      if major=='3':
        conf.write('#define SLEPC_HAVE_PRIMME3 1\n')
    except: pass

