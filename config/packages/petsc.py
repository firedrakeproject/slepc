#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain
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

import package, os, sys, commands

class PETSc(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename = 'petsc'

  def Check(self):
    self.havepackage = self.Link([],[],[])

  def InitDir(self,prefixdir):
    if 'PETSC_DIR' in os.environ:
      self.dir = os.environ['PETSC_DIR']
      if not os.path.exists(self.dir):
        sys.exit('ERROR: PETSC_DIR enviroment variable is not valid')
    else:
      if prefixdir:
        self.dir = prefixdir
        os.environ['PETSC_DIR'] = self.dir
      else:
        sys.exit('ERROR: PETSC_DIR enviroment variable is not set')

  def LoadVersion(self):
    try:
      f = open(os.path.join(self.dir,'include','petscversion.h'))
      for l in f.readlines():
        l = l.split()
        if len(l) == 3:
          if l[1] == 'PETSC_VERSION_RELEASE':
            self.release = l[2]
          if l[1] == 'PETSC_VERSION_MAJOR':
            major = l[2]
          elif l[1] == 'PETSC_VERSION_MINOR':
            minor = l[2]
          elif l[1] == 'PETSC_VERSION_SUBMINOR':
            subminor = l[2]
          elif l[1] == 'PETSC_VERSION_PATCH':
            patchlevel = l[2]
      f.close()
      self.version = major + '.' + minor
      self.lversion = major + '.' + minor + '.' + subminor
    except:
      self.log.Exit('ERROR: File error while reading PETSc version')

    # Check whether this is a working copy of the repository
    self.isrepo = False
    if os.path.exists(os.path.join(self.dir,'.git')):
      (status, output) = commands.getstatusoutput('cd '+self.dir+';git rev-parse')
      if not status:
        self.isrepo = True
        (status, self.gitrev) = commands.getstatusoutput('cd '+self.dir+';git log -1 --pretty=format:%H')
        (status, self.gitdate) = commands.getstatusoutput('cd '+self.dir+';git log -1 --pretty=format:%ci')
        (status, self.branch) = commands.getstatusoutput('cd '+self.dir+';git describe --contains --all HEAD')

  def LoadConf(self):
    if 'PETSC_ARCH' in os.environ and os.environ['PETSC_ARCH']:
      self.isinstall = False
      self.arch = os.environ['PETSC_ARCH']
      petscvariables = os.path.join(self.dir,self.arch,'lib','petsc','conf','petscvariables')
      petscconf_h = os.path.join(self.dir,self.arch,'include','petscconf.h')
    else:
      self.isinstall = True
      petscvariables = os.path.join(self.dir,'lib','petsc','conf','petscvariables')
      petscconf_h = os.path.join(self.dir,'include','petscconf.h')

    self.build_using_cmake = 0
    self.make_is_gnumake = 0
    self.language = 'c'
    self.bfort = 'nobfortinpetsc'
    try:
      f = open(petscvariables)
      for l in f.readlines():
        r = l.split('=',1)
        if len(r)!=2: continue
        k = r[0].strip()
        v = r[1].strip()
        if k == 'PETSC_SCALAR':
          self.scalar = v
        elif k == 'PETSC_PRECISION':
          self.precision = v
        elif k == 'MAKE':
          self.make = v
        elif k == 'DESTDIR':
          self.destdir = v
        elif k == 'BFORT':
          self.bfort = v
        elif k == 'TEST_RUNS':
          self.test_runs = v
        elif k == 'CC':
          self.cc = v
        elif k == 'CC_FLAGS':
          self.cc_flags = v
        elif k == 'FC' and not v=='':
          self.fc = v
        elif k == 'FC_FLAGS':
          self.fc_flags = v
        elif k == 'AR':
          self.ar = v
        elif k == 'AR_FLAGS':
          self.ar_flags = v
        elif k == 'AR_LIB_SUFFIX':
          self.ar_lib_suffix = v
        elif k == 'CC_LINKER_SLFLAG':
          self.slflag = v
        elif k == 'RANLIB':
          self.ranlib = v
        elif k == 'PETSC_BUILD_USING_CMAKE':
          self.build_using_cmake = v
        elif k == 'MAKE_IS_GNUMAKE':
          self.make_is_gnumake = v
        elif k == 'PETSC_LANGUAGE' and v=='CXXONLY':
          self.language = 'c++'
      f.close()
    except:
      self.log.Exit('ERROR: cannot process file ' + petscvariables)

    self.ind64 = False
    self.mpiuni = False
    self.debug = False
    self.singlelib = False
    self.blaslapackunderscore = False
    self.blaslapackint64 = False
    try:
      f = open(petscconf_h)
      for l in f.readlines():
        l = l.split()
        if len(l)==3 and l[0]=='#define' and l[1]=='PETSC_USE_64BIT_INDICES' and l[2]=='1':
          self.ind64 = True
        elif len(l)==3 and l[0]=='#define' and l[1]=='PETSC_HAVE_MPIUNI' and l[2]=='1':
          self.mpiuni = True
        elif len(l)==3 and l[0]=='#define' and l[1]=='PETSC_USE_DEBUG' and l[2]=='1':
          self.debug = True
        elif len(l)==3 and l[0]=='#define' and l[1]=='PETSC_USE_SINGLE_LIBRARY' and l[2]=='1':
          self.singlelib = True
        elif len(l)==3 and l[0]=='#define' and l[1]=='PETSC_BLASLAPACK_UNDERSCORE' and l[2]=='1':
          self.blaslapackunderscore = True
        elif len(l)==3 and l[0]=='#define' and l[1]=='HAVE_64BIT_BLAS_INDICES' and l[2]=='1':
          self.blaslapackint64 = True
        elif self.isinstall and len(l)==3 and l[0]=='#define' and l[1]=='PETSC_ARCH':
          self.arch = l[2].strip('"')
      f.close()
    except:
      if self.isinstall:
        self.log.Exit('ERROR: cannot process file ' + petscconf_h + ', maybe you forgot to set PETSC_ARCH')
      else:
        self.log.Exit('ERROR: cannot process file ' + petscconf_h)

    # empty PETSC_ARCH, guess an arch name
    if self.isinstall and not self.arch:
      self.arch = 'arch-' + sys.platform.replace('cygwin','mswin')+ '-' + self.language
      if self.debug:
        self.arch += '-debug'
      else:
        self.arch += '-opt'
      if not 'real' in self.scalar:
        self.arch += '-' + self.scalar

