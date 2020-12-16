#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os,shutil,log,package
try:
  from urllib import urlretrieve
except ImportError:
  from urllib.request import urlretrieve

class HPDDM(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename    = 'hpddm'
    self.downloadable   = True
    self.gitcommit      = 'd29735a339b1dcb939a0e5aaab60271c7537ee89'
    self.archive        = self.gitcommit+'.tar.gz'
    self.url            = 'https://github.com/hpddm/hpddm/archive/'+self.archive
    self.dirname        = 'hpddm-'+self.gitcommit
    self.supportssingle = True
    self.supports64bint = True
    self.ProcessArgs(argdb)

  def Precondition(self,slepc,petsc):
    pkg = self.packagename.upper()
    if not petsc.cxxdialectcxx11:
      self.log.Exit(pkg+' requires C++11')
    if not petsc.buildsharedlib:
      self.log.Exit(pkg+' requires a shared library build')
    if self.packagename in petsc.packages:
      self.log.Exit(pkg+' requires PETSc to be built without '+pkg)
    package.Package.Precondition(self,slepc,petsc)

  def DownloadAndInstall(self,slepcconf,slepcvars,slepc,petsc,archdir,prefixdir):
    externdir = slepc.CreateDir(archdir,'externalpackages')
    builddir  = self.Download(externdir,slepc.downloaddir)
    incdir,libdir = slepc.CreatePrefixDirs(prefixdir)
    cont  = 'include '+os.path.join(petsc.dir,petsc.arch,'lib','petsc','conf','petscvariables')+'\n'
    cont += 'soname:\n'
    cont += '\t@echo $(call SONAME_FUNCTION,'+os.path.join(libdir,'libhpddm_petsc')+',0)\n'
    cont += 'sl_linker:\n'
    cont += '\t@echo $(call SL_LINKER_FUNCTION,'+os.path.join(libdir,'libhpddm_petsc')+',0,0)\n'
    self.log.write('Using makefile:\n')
    self.log.write(cont)
    mfile = open(os.path.join(builddir,'SONAME_SL_LINKER'),'w')
    mfile.write(cont)
    mfile.close()
    d = os.path.join(petsc.dir,petsc.arch,'lib')
    l = petsc.slflag+d+' -L'+d+' -lpetsc'
    d = libdir
    if petsc.isinstall:
      branch = 'current'
      if slepc.isrepo and slepc.branch != 'release':
        branch = 'master'
      urlretrieve('https://www.mcs.anl.gov/petsc/petsc-'+branch+'/src/ksp/ksp/impls/hpddm/hpddm.cxx',os.path.join(builddir,'interface','ksphpddm.cxx'));
      urlretrieve('https://www.mcs.anl.gov/petsc/petsc-'+branch+'/src/ksp/pc/impls/hpddm/hpddm.cxx',os.path.join(builddir,'interface','pchpddm.cxx'));
    else:
      shutil.copyfile(os.path.join(petsc.dir,'src','ksp','ksp','impls','hpddm','hpddm.cxx'),os.path.join(builddir,'interface','ksphpddm.cxx'))
      shutil.copyfile(os.path.join(petsc.dir,'src','ksp','pc','impls','hpddm','hpddm.cxx'),os.path.join(builddir,'interface','pchpddm.cxx'))
    cmd = petsc.cxx+' '+petsc.cxx_flags+' -I'+os.path.join('.','include')+' -I'+os.path.join(petsc.dir,petsc.arch,'include')+' -I'+os.path.join(slepc.dir,'include')+' -I'+os.path.join(archdir,'include')+' -I'+os.path.join(petsc.dir,'include')+' -DPETSC_HAVE_SLEPC=1 -DSLEPC_LIB_DIR="'+d+'"'
    (result,output) = self.RunCommand('cd '+builddir+'&&'+cmd+' '+os.path.join('interface','ksphpddm.cxx')+' -c -o '+os.path.join('interface','ksphpddm.o')+'&&'+cmd+' '+os.path.join('interface','pchpddm.cxx')+' -c -o '+os.path.join('interface','pchpddm.o')+'&&'+cmd+' '+os.path.join('interface','hpddm_petsc.cpp')+' -c -o '+os.path.join('interface','hpddm_petsc.o'))
    if result:
      self.log.Exit('Compilation of HPDDM failed')
    (result,output) = self.RunCommand('cd '+builddir+'&& make -f SONAME_SL_LINKER soname && make -f SONAME_SL_LINKER sl_linker')
    if result:
      self.log.Exit('Calling PETSc SONAME_FUNCTION or SL_LINKER_FUNCTION failed')
    lines = output.splitlines()
    (result,output) = self.RunCommand('cd '+builddir+'&& '+petsc.cxx+' '+petsc.cxx_flags+' '+os.path.join('interface','hpddm_petsc.o')+' '+os.path.join('interface','pchpddm.o')+' '+os.path.join('interface','ksphpddm.o')+' -o '+lines[0]+' '+lines[1]+' '+l+' && ln -sf '+lines[0]+' '+os.path.join(d,'libhpddm_petsc.'+petsc.sl_suffix))
    if result:
      self.log.Exit('Installation of HPDDM failed')
    for root,dirs,files in os.walk(os.path.join(builddir,'include')):
      for name in files:
        shutil.copyfile(os.path.join(builddir,'include',name),os.path.join(incdir,name))
    l = petsc.slflag+d+' -L'+d+' -lhpddm_petsc'
    f = '-I'+incdir
    # Write configuration files
    slepcconf.write('#define SLEPC_HAVE_HPDDM 1\n')
    slepcvars.write('HPDDM_LIB = '+l+'\n')
    slepcvars.write('HPDDM_INCLUDE = '+f+'\n')
    self.packageflags = [l] + [f]
    self.havepackage = True

