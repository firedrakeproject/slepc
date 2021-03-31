#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os, shutil, log, package

class Evsl(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename    = 'evsl'
    self.installable    = True
    self.downloadable   = True
    self.gitcommit      = '814fc8251c453b999b6f33ccff48c728104597aa'  #master 27-mar-2021
    self.url            = 'https://github.com/eigs/EVSL/archive/'+self.gitcommit+'.tar.gz'
    self.archive        = 'evsl-'+self.gitcommit+'.tar.gz'
    self.dirname        = 'evsl-'+self.gitcommit
    self.supportsscalar = ['real']
    self.hasheaders     = True
    self.ProcessArgs(argdb)


  def SampleCode(self,petsc):
    code =  '#include <evsl.h>\n'
    code += 'int main() {\n'
    code += '  double xintv[4],tol=0.0,*vinit=NULL,*lam=NULL,*Y=NULL,*res=NULL;\n'
    code += '  int mlan=0,nev2,ierr;\n'
    code += '  polparams pol;\n'
    code += '  EVSLStart();\n'
    code += '  ierr = ChebLanNr(xintv, mlan, tol, vinit, &pol, &nev2, &lam, &Y, &res, NULL);\n'
    code += '  EVSLFinish();\n'
    code += '  return ierr;\n}\n'
    return code


  def Check(self,slepcconf,slepcvars,petsc,archdir):
    code = self.SampleCode(petsc)
    if self.packagedir:
      dirs = [os.path.join(self.packagedir,'lib'),self.packagedir,os.path.join(self.packagedir,'lib64')]
      incdirs = [os.path.join(self.packagedir,'include'),self.packagedir]
    else:
      dirs = self.GenerateGuesses('Evsl',archdir)
      incdirs = self.GenerateGuesses('Evsl',archdir,'include')

    libs = self.packagelibs
    if not libs:
      libs = ['-levsl']
    includes = self.packageincludes
    if not includes:
      includes = ['.']

    for (d,i) in zip(dirs,incdirs):
      if d:
        if petsc.buildsharedlib:
          l = [petsc.slflag + d] + ['-L' + d] + libs
        else:
          l = ['-L' + d] + libs
        f = ['-I' + i]
      else:
        l = libs
        f = ['-I' + includes[0]]
      result = self.Link([],[],l+f,code,' '.join(f),petsc.language)
      if result:
        slepcconf.write('#define SLEPC_HAVE_EVSL 1\n')
        slepcvars.write('EVSL_LIB = ' + ' '.join(l) + '\n')
        slepcvars.write('EVSL_INCLUDE = ' + ' '.join(f) + '\n')
        self.havepackage = True
        self.packageflags = l+f
        return

    self.log.Exit('Unable to link with EVSL library in directories'+' '.join(dirs)+' with libraries and link flags '+' '.join(libs)+' [NOTE: make sure EVSL version is 1.1.1 at least]')


  def DownloadAndInstall(self,slepcconf,slepcvars,slepc,petsc,archdir,prefixdir):
    externdir = slepc.CreateDir(archdir,'externalpackages')
    builddir  = os.path.join(self.Download(externdir,slepc.downloaddir),'EVSL_1.1.1')
    incdir,libdir = slepc.CreatePrefixDirs(prefixdir)

    # Build package
    confopt = '--prefix='+prefixdir+' --with-blas-lib="'+petsc.blaslapack_lib+'" --with-lapack-lib="'+petsc.blaslapack_lib+'" CC="'+petsc.cc+'" CFLAGS="'+petsc.getCFlags()+'" F77="'+petsc.fc+'" FFLAGS="'+petsc.getFFlags()+'" FC="'+petsc.fc+'" FCFLAGS="'+petsc.getFFlags()+'"'
    if petsc.buildsharedlib:
      confopt = confopt+' --enable-shared'
    if 'mkl' in petsc.packages and 'mkl_pardiso' in petsc.packages and 'MKLROOT' in os.environ:
      confopt = confopt+' --with-mkl-pardiso --with-intel-mkl'
    (result,output) = self.RunCommand('cd '+builddir+'&& ./configure '+confopt+' && '+petsc.make+' && '+petsc.make+' install')
    if result:
      self.log.Exit('Installation of EVSL failed')

    if petsc.buildsharedlib:
      l = petsc.slflag + libdir + ' -L' + libdir + ' -levsl'
    else:
      l = '-L' + libdir + ' -levsl'
    f = '-I' + incdir

    # Check build
    code = self.SampleCode(petsc)
    (result, output) = self.Link([],[],[l]+[f],code,f,petsc.language)
    if not result:
      self.log.Exit('Unable to link with downloaded EVSL')

    # Write configuration files
    slepcconf.write('#define SLEPC_HAVE_EVSL 1\n')
    slepcvars.write('EVSL_LIB = ' + l + '\n')
    slepcvars.write('EVSL_INCLUDE = ' + f + '\n')

    self.havepackage = True
    self.packageflags = [l] + [f]

