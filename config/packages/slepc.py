from __future__ import print_function
#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import argdb, os, sys, package

class SLEPc(package.Package):

  def __init__(self,argdb,log):
    self.log         = log
    self.clean       = argdb.PopBool('with-clean')[0]
    self.prefixdir   = argdb.PopPath('prefix')[0]
    self.isinstall   = not self.prefixdir==''
    self.datadir     = argdb.PopPath('DATAFILESPATH',exist=True)[0]
    self.downloaddir = argdb.PopPath('with-packages-download-dir',exist=True)[0]

  def ShowHelp(self):
    wd = package.Package.wd
    print('SLEPc:')
    print('  --with-clean=<bool>'.ljust(wd)+': Delete prior build files including externalpackages')
    print('  --prefix=<dir>'.ljust(wd)+': Specify location to install SLEPc (e.g., /usr/local)')
    print('  --DATAFILESPATH=<dir>'.ljust(wd)+': Specify location of datafiles (for SLEPc developers)')
    print('  --with-packages-download-dir=<dir>'.ljust(wd)+': Skip network download of tarballs and locate them in specified dir')

  def InitDir(self):
    if 'SLEPC_DIR' in os.environ:
      self.dir = os.environ['SLEPC_DIR']
      if not os.path.exists(self.dir) or not os.path.exists(os.path.join(self.dir,'config')):
        self.log.Exit('SLEPC_DIR enviroment variable is not valid')
      if os.path.realpath(os.getcwd()) != os.path.realpath(self.dir):
        self.log.Exit('SLEPC_DIR is not the current directory')
    else:
      self.dir = os.getcwd()
      if not os.path.exists(os.path.join(self.dir,'config')):
        self.log.Exit('Current directory is not valid')

  def LoadVersion(self):
    try:
      f = open(os.path.join(self.dir,'include','slepcversion.h'))
      for l in f.readlines():
        l = l.split()
        if len(l) == 3:
          if l[1] == 'SLEPC_VERSION_RELEASE':
            self.release = l[2]
          if l[1] == 'SLEPC_VERSION_MAJOR':
            major = l[2]
          elif l[1] == 'SLEPC_VERSION_MINOR':
            minor = l[2]
          elif l[1] == 'SLEPC_VERSION_SUBMINOR':
            subminor = l[2]
      f.close()
      self.version = major + '.' + minor
      self.lversion = major + '.' + minor + '.' + subminor
      self.nversion = int(major)*100 + int(minor)
    except:
      self.log.Exit('File error while reading SLEPc version')

    # Check whether this is a working copy of the repository
    self.isrepo = False
    if os.path.exists(os.path.join(self.dir,'src','docs')):
      self.isrepo = True
      (status, output) = self.RunCommand('git rev-parse')
      if status:
        self.log.Warn('SLEPC_DIR appears to be a git working copy, but git is not found in PATH')
        self.gitrev = 'N/A'
        self.gitdate = 'N/A'
        self.branch = 'N/A'
      else:
        (status, self.gitrev) = self.RunCommand('git describe')
        if not self.gitrev:
          (status, self.gitrev) = self.RunCommand('git log -1 --pretty=format:%H')
        (status, self.gitdate) = self.RunCommand('git log -1 --pretty=format:%ci')
        (status, self.branch) = self.RunCommand('git describe --contains --all HEAD')

