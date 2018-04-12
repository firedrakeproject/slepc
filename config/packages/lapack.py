#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

import log, package

class Lapack(package.Package):

  def __init__(self,argdb,log):
    package.Package.__init__(self,argdb,log)
    self.packagename = 'lapack'

  def ShowInfo(self):
    if hasattr(self,'missing'):
      self.log.Println('LAPACK missing functions:')
      self.log.Print('  ')
      for i in self.missing: self.log.Print(i)
      self.log.Println('')
      self.log.Println('')
      self.log.Println('WARNING: Some SLEPc functionality will not be available')
      self.log.Println('PLEASE reconfigure and recompile PETSc with a full LAPACK implementation')

  def Process(self,conf,vars,cmake,petsc,archdir=''):
    self.make = petsc.make
    if petsc.buildsharedlib:
      self.slflag = petsc.slflag
    self.log.NewSection('Checking LAPACK library...')
    self.Check(conf,vars,cmake,petsc)

  def Check(self,conf,vars,cmake,petsc):

    # LAPACK standard functions
    l = ['laev2','gehrd','lanhs','lange','trexc','trevc','geevx','gees','ggev','ggevx','gelqf','geqp3','gesdd','tgexc','tgevc','pbtrf','stedc','hsein','larfg','larf','lacpy','lascl','lansy','laset','trsyl','trtri']

    # LAPACK functions with different real and complex versions
    if petsc.scalar == 'real':
      l += ['orghr','syevr','syevd','sytrd','sygvd','ormlq','orgtr']
      if petsc.precision == 'single':
        prefix = 's'
      elif petsc.precision == '__float128':
        prefix = 'q'
      else:
        prefix = 'd'
    else:
      l += ['unghr','heevr','heevd','hetrd','hegvd','unmlq','ungtr']
      if petsc.precision == 'single':
        prefix = 'c'
      elif petsc.precision == '__float128':
        prefix = 'w'
      else:
        prefix = 'z'

    # add prefix to LAPACK names
    functions = []
    for i in l:
      functions.append(prefix + i)

    # in this case, the real name represents both versions
    namesubst = {'unghr':'orghr', 'heevr':'syevr', 'heevd':'syevd', 'hetrd':'sytrd', 'hegvd':'sygvd', 'unmlq':'ormlq', 'ungtr':'orgtr'}

    # LAPACK functions which are always used in real version
    if petsc.precision == 'single':
      functions += ['sstevr','sbdsdc','slamch','slag2','slasv2','slartg','slaln2','slaed4','slamrg','slapy2']
    elif petsc.precision == '__float128':
      functions += ['qstevr','qbdsdc','qlamch','qlag2','qlasv2','qlartg','qlaln2','qlaed4','qlamrg','qlapy2']
    else:
      functions += ['dstevr','dbdsdc','dlamch','dlag2','dlasv2','dlartg','dlaln2','dlaed4','dlamrg','dlapy2']

    # check for all functions at once
    all = []
    for i in functions:
      f =  '#if defined(PETSC_BLASLAPACK_UNDERSCORE)\n'
      f += i + '_\n'
      f += '#elif defined(PETSC_BLASLAPACK_CAPS) || defined(PETSC_BLASLAPACK_STDCALL)\n'
      f += i.upper() + '\n'
      f += '#else\n'
      f += i + '\n'
      f += '#endif\n'
      all.append(f)

    self.log.write('=== Checking all LAPACK functions...')
    if self.Link(all,[],[]):
      return

    # check functions one by one
    self.missing = []
    for i in functions:
      f =  '#if defined(PETSC_BLASLAPACK_UNDERSCORE)\n'
      f += i + '_\n'
      f += '#elif defined(PETSC_BLASLAPACK_CAPS) || defined(PETSC_BLASLAPACK_STDCALL)\n'
      f += i.upper() + '\n'
      f += '#else\n'
      f += i + '\n'
      f += '#endif\n'

      self.log.write('=== Checking LAPACK '+i+' function...')
      if not self.Link([f],[],[]):
        self.missing.append(i)
        # some complex functions are represented by their real names
        if i[1:] in namesubst:
          nf = namesubst[i[1:]]
        else:
          nf = i[1:]
        conf.write('#ifndef SLEPC_MISSING_LAPACK_' + nf.upper() + '\n#define SLEPC_MISSING_LAPACK_' + nf.upper() + ' 1\n#endif\n\n')
        cmake.write('set (SLEPC_MISSING_LAPACK_' + nf.upper() + ' YES)\n')

    if self.missing:
      cmake.write('mark_as_advanced (' + ' '.join([s.upper() for s in self.missing]) + ')\n')

