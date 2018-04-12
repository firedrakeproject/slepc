from __future__ import print_function
#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import os, sys, commands, tempfile, shutil, urllib, urlparse, tarfile
import log, argdb

class Package:

  def __init__(self,argdb,log):
    self.installable     = False
    self.downloadable    = False
    self.downloadpackage = 0
    self.packagedir      = ''
    self.packagelibs     = []
    self.packageurl      = ''
    self.log             = log
    self.supportsscalar  = ['real', 'complex']
    self.supportssingle  = False
    self.supports64bint  = False
    self.fortran         = False

  def ProcessArgs(self,argdb):
    self.requested = False
    self.havepackage = False
    if self.installable:
      string,found = argdb.PopPath('with-'+self.packagename+'-dir')
      if found:
        self.requested = True
        self.packagedir = string
      string,found = argdb.PopString('with-'+self.packagename+'-flags')
      if found:
        self.requested = True
        self.packagelibs = string.split(',')
      if argdb.PopBool('with-'+self.packagename):
        self.requested = True
    if self.downloadable:
      url,flag,found = argdb.PopUrl('download-'+self.packagename)
      if found:
        if self.requested:
          self.log.Exit('ERROR: Cannot request both download and install simultaneously')
        self.requested = True
        self.download = True
        self.packageurl = url
        self.downloadpackage = flag

  def Process(self,conf,vars,cmake,petsc,archdir=''):
    self.make = petsc.make
    if petsc.buildsharedlib:
      self.slflag = petsc.slflag
    if self.requested:
      name = self.packagename.upper()
      if self.downloadpackage:
        if hasattr(self,'version'):
          self.log.NewSection('Installing '+name+' version '+self.version+'...')
        else:
          self.log.NewSection('Installing '+name+'...')
        self.Precondition(petsc)
        self.Install(conf,vars,cmake,petsc,archdir)
      elif self.installable:
        self.log.NewSection('Checking '+name+'...')
        self.Precondition(petsc)
        self.Check(conf,vars,cmake,petsc)

  def Precondition(self,petsc):
    package = self.packagename.upper()
    if petsc.scalar == 'complex':
      if 'complex' not in self.supportsscalar:
        self.log.Exit('ERROR: '+package+' does not support complex scalars.')
    elif petsc.scalar == 'real':
      if 'real' not in self.supportsscalar:
        self.log.Exit('ERROR: '+package+' is supported only with complex scalars.')
    if petsc.precision == 'single':
      if not self.supportssingle:
        self.log.Exit('ERROR: '+package+' is supported only in double precision.')
    elif petsc.precision != 'double':
      self.log.Exit('ERROR: precision '+petsc.precision+' is not supported for external packages.')
    if petsc.ind64 and not self.supports64bint:
      self.log.Exit('ERROR: '+package+' cannot be used with 64-bit integers.')
    if self.downloadpackage and self.fortran and not hasattr(petsc,'fc'):
      self.log.Exit('ERROR: option --download-'+self.packagename+' requires a Fortran compiler.')

  def Download(self,externdir,builddir,prefix=None):
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
        url = self.url
      localFile = os.path.join(externdir,self.archive)
      self.log.write('Downloading '+url+' to '+localFile)

      if os.path.exists(localFile):
        os.remove(localFile)
      try:
        urllib.urlretrieve(url, localFile)
      except Exception as e:
        filename = os.path.basename(urlparse.urlparse(url)[2])
        failureMessage = '''\
Unable to download package %s from: %s
* If your network is disconnected - please reconnect and rerun ./configure
* Alternatively, you can download the above URL manually, to /yourselectedlocation/%s
  and use the configure option:
  --download-%s=/yourselectedlocation/%s
''' % (self.packagename, url, filename, self.packagename, filename)
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
          tar = tarfile.open(localFile, 'r:gz')
          tar.extractall(path=externdir)
          tar.close()
          os.remove(localFile)
        else:
          result,output = commands.getstatusoutput('cd '+externdir+'; gunzip '+self.archive+'; tar -xf '+self.archive.split('.gz')[0])
          os.remove(localFile.split('.gz')[0])
      except RuntimeError as e:
        self.log.Exit('Error uncompressing '+self.archive+': '+str(e))

      # Rename directory
      if prefix is not None:
        for filename in os.listdir(externdir):
          if filename.startswith(prefix):
            os.rename(os.path.join(externdir,filename),builddir)

  def ShowHelp(self):
    wd = 31
    if self.downloadable or self.installable:
      print(self.packagename.upper()+':')
    if self.downloadable:
      print(('  --download-'+self.packagename+'[=<fname>]').ljust(wd)+': Download and install '+self.packagename.upper()+' in SLEPc directory')
    if self.installable:
      print(('  --with-'+self.packagename+'=<bool>').ljust(wd)+': Indicate if you wish to test for '+self.packagename.upper())
      print(('  --with-'+self.packagename+'-dir=<dir>').ljust(wd)+': Indicate the directory for '+self.packagename.upper()+' libraries')
      print(('  --with-'+self.packagename+'-flags=<flags>').ljust(wd)+': Indicate comma-separated flags for linking '+self.packagename.upper())

  def ShowInfo(self):
    if self.havepackage:
      self.log.Println(self.packagename.upper()+' library flags:')
      self.log.Println(' '+' '.join(self.packageflags))

  def TestRuns(self,petsc):
    if self.havepackage:
      return [self.packagename.upper()]
    else:
      return []

  def LinkWithOutput(self,functions,callbacks,flags):

    # Create temporary directory and makefile
    try:
      tmpdir = tempfile.mkdtemp(prefix='slepc-')
      if not os.path.isdir(tmpdir): os.mkdir(tmpdir)
    except:
      self.log.Exit('ERROR: Cannot create temporary directory')
    try:
      makefile = open(os.path.join(tmpdir,'makefile'),'w')
      makefile.write('checklink: checklink.o chkopts\n')
      makefile.write('\t${CLINKER} -o checklink checklink.o ${TESTFLAGS} ${PETSC_SNES_LIB}\n')
      makefile.write('\t@${RM} -f checklink checklink.o\n')
      makefile.write('LOCDIR = ./\n')
      makefile.write('include '+os.path.join('${PETSC_DIR}','lib','petsc','conf','variables')+'\n')
      makefile.write('include '+os.path.join('${PETSC_DIR}','lib','petsc','conf','rules')+'\n')
      makefile.close()
    except:
      self.log.Exit('ERROR: Cannot create makefile in temporary directory')

    # Create source file
    code = '#include "petscsnes.h"\n'
    for f in functions:
      code += 'PETSC_EXTERN int\n' + f + '();\n'

    for c in callbacks:
      code += 'int '+ c + '() { return 0; } \n'

    code += 'int main() {\n'
    code += 'Vec v; Mat m; KSP k;\n'
    code += 'PetscInitializeNoArguments();\n'
    code += 'VecCreate(PETSC_COMM_WORLD,&v);\n'
    code += 'MatCreate(PETSC_COMM_WORLD,&m);\n'
    code += 'KSPCreate(PETSC_COMM_WORLD,&k);\n'
    for f in functions:
      code += f + '();\n'
    code += 'return 0;\n}\n'

    cfile = open(os.path.join(tmpdir,'checklink.c'),'w')
    cfile.write(code)
    cfile.close()

    # Try to compile test program
    (result, output) = commands.getstatusoutput('cd ' + tmpdir + ';' + self.make + ' checklink TESTFLAGS="'+' '.join(flags)+'"')
    shutil.rmtree(tmpdir)

    if result:
      return (0,code + output)
    else:
      return (1,code + output)

  def Link(self,functions,callbacks,flags):
    (result, output) = self.LinkWithOutput(functions,callbacks,flags)
    self.log.write(output)
    return result

  def FortranLink(self,functions,callbacks,flags):
    output = '\n=== With linker flags: '+' '.join(flags)

    f = []
    for i in functions:
      f.append(i+'_')
    c = []
    for i in callbacks:
      c.append(i+'_')
    (result, output1) = self.LinkWithOutput(f,c,flags)
    output1 = '\n====== With underscore Fortran names\n' + output1
    if result: return ('UNDERSCORE',output1)

    f = []
    for i in functions:
      f.append(i.upper())
    c = []
    for i in callbacks:
      c.append(i.upper())
    (result, output2) = self.LinkWithOutput(f,c,flags)
    output2 = '\n====== With capital Fortran names\n' + output2
    if result: return ('CAPS',output2)

    (result, output3) = self.LinkWithOutput(functions,callbacks,flags)
    output3 = '\n====== With unmodified Fortran names\n' + output3
    if result: return ('STDCALL',output3)

    return ('',output + output1 + output2 + output3)

  def GenerateGuesses(self,name):
    installdirs = [os.path.join(os.path.sep,'usr','local'),os.path.join(os.path.sep,'opt')]
    if 'HOME' in os.environ:
      installdirs.insert(0,os.environ['HOME'])

    dirs = []
    for i in installdirs:
      dirs = dirs + [os.path.join(i,'lib')]
      for d in [name,name.upper(),name.lower()]:
        dirs = dirs + [os.path.join(i,d)]
        dirs = dirs + [os.path.join(i,d,'lib')]
        dirs = dirs + [os.path.join(i,'lib',d)]

    for d in dirs[:]:
      if not os.path.exists(d):
        dirs.remove(d)
    dirs = [''] + dirs
    return dirs

  def FortranLib(self,conf,vars,cmake,dirs,libs,functions,callbacks = []):
    name = self.packagename.upper()
    error = ''
    mangling = ''
    for d in dirs:
      for l in libs:
        if d:
          if hasattr(self,'slflag'):
            flags = [self.slflag + d] + ['-L' + d] + l
          else:
            flags = ['-L' + d] + l
        else:
          flags = l
        (mangling, output) = self.FortranLink(functions,callbacks,flags)
        error += output
        if mangling: break
      if mangling: break

    if mangling:
      self.log.write(output)
    else:
      self.log.write(error)
      self.log.Println('\nERROR: Unable to link with library '+ name)
      self.log.Println('ERROR: In directories '+' '.join(dirs))
      self.log.Println('ERROR: With flags '+' '.join(flags))
      self.log.Exit('')

    conf.write('#ifndef SLEPC_HAVE_' + name + '\n#define SLEPC_HAVE_' + name + ' 1\n#define SLEPC_' + name + '_HAVE_'+mangling+' 1\n#endif\n\n')
    vars.write(name + '_LIB = '+' '.join(flags)+'\n')
    cmake.write('set (SLEPC_HAVE_' + name + ' YES)\n')
    libname = ' '.join([s.lstrip('-l') for s in l])
    cmake.write('set (' + name + '_LIB "")\nforeach (libname ' + libname + ')\n  string (TOUPPER ${libname} LIBNAME)\n  find_library (${LIBNAME}LIB ${libname} HINTS '+ d +')\n  list (APPEND ' + name + '_LIB "${${LIBNAME}LIB}")\nendforeach()\n')

    self.havepackage = True
    self.packageflags = flags

