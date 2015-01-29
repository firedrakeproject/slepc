#!/usr/bin/env python

"""
SLEPc: Scalable Library for Eigenvalue Problem Computations
===========================================================

SLEPc is a software library for the solution of large scale sparse
eigenvalue problems on parallel computers. It is an extension of PETSc
and can be used for either standard or generalized eigenproblems, with
real or complex arithmetic. It can also be used for computing a
partial SVD of a large, sparse, rectangular matrix, and to solve
nonlinear eigenvalue problems

.. note::

   To install ``PETSc``, ``SLEPc``, ``petsc4py``, and ``slepc4py``
   (``mpi4py`` is optional but highly recommended) use::

     $ pip install numpy mpi4py
     $ pip install petsc petsc4py
     $ pip install slepc slepc4py

.. tip::

  You can also install the in-development versions with::

    $ pip install Cython numpy mpi4py
    $ pip install --no-deps https://bitbucket.org/petsc/petsc/get/master.tar.gz
    $ pip install --no-deps https://bitbucket.org/petsc/petsc4py/get/master.tar.gz
    $ pip install --no-deps https://bitbucket.org/slepc/slepc/get/master.tar.gz
    $ pip install --no-deps https://bitbucket.org/slepc/slepc4py/get/master.tar.gz

"""

import sys, os
from distutils.core import setup
from distutils.util import get_platform
from distutils.spawn import find_executable
from distutils.command.build import build as _build
if 'setuptools' in sys.modules:
    from setuptools.command.install import install as _install
else:
    from distutils.command.install import install as _install
from distutils.command.sdist import sdist as _sdist
from distutils import log

init_py = """\
# Author:  SLEPc Team
# Contact: slepc-maint@upv.es

def get_slepc_dir():
    import os
    return os.path.dirname(__file__)

def get_config():
    conf = {}
    conf['SLEPC_DIR'] = get_slepc_dir()
    return conf
"""

metadata = {
    'provides' : ['slepc'],
    'requires' : [],
}

def bootstrap():
    from os.path import join, isdir, abspath
    # Set SLEPC_DIR
    SLEPC_DIR  = abspath(os.getcwd())
    os.environ['SLEPC_DIR']  = SLEPC_DIR
    # Check PETSC_DIR/PETSC_ARCH
    PETSC_DIR  = os.environ.get('PETSC_DIR',  "")
    PETSC_ARCH = os.environ.get('PETSC_ARCH', "")
    if not (PETSC_DIR and isdir(PETSC_DIR)):
        PETSC_DIR = None
        try: del os.environ['PETSC_DIR']
        except KeyError: pass
        PETSC_ARCH = None
        try: del os.environ['PETSC_ARCH']
        except KeyError: pass
    elif not (PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH))):
        PETSC_ARCH = None
        try: del os.environ['PETSC_ARCH']
        except KeyError: pass
    # Generate package __init__.py file
    from distutils.dir_util import mkpath
    pkgdir = os.path.join(SLEPC_DIR, 'pypi')
    pkgfile = os.path.join(pkgdir, '__init__.py')
    if not os.path.exists(pkgdir): mkpath(pkgdir)
    fh = open(pkgfile, 'wt')
    fh.write(init_py)
    fh.close()
    if ('setuptools' in sys.modules):
        metadata['zip_safe'] = False
        if not PETSC_DIR:
            metadata['install_requires']= ['petsc>=3.5,<3.6']

def get_petsc_dir():
    PETSC_DIR = os.environ.get('PETSC_DIR')
    if PETSC_DIR: return PETSC_DIR
    try:
        import petsc
        PETSC_DIR = petsc.get_petsc_dir()
    except ImportError:
        log.warn("PETSC_DIR not specified")
        PETSC_DIR = os.path.join(os.path.sep, 'usr', 'local', 'petsc')
    return PETSC_DIR

def get_petsc_arch():
    PETSC_ARCH = os.environ.get('PETSC_ARCH')
    return PETSC_ARCH

def config(dry_run=False):
    log.info('SLEPc: configure')
    if dry_run: return
    # PETSc
    # Run SLEPc configure
    os.environ['PETSC_DIR'] = get_petsc_dir()
    status = os.system(" ".join((
            find_executable('python'),
            os.path.join('config', 'configure.py'),
            )))
    if status != 0: raise RuntimeError(status)

def build(dry_run=False):
    log.info('SLEPc: build')
    if dry_run: return
    # Run SLEPc build
    PETSC_ARCH = get_petsc_arch() or ''
    if PETSC_ARCH: PETSC_ARCH = 'PETSC_ARCH=' + PETSC_ARCH
    status = os.system(" ".join((
            find_executable('make'),
            'PETSC_DIR='+get_petsc_dir(), PETSC_ARCH,
            'all',
            )))
    if status != 0: raise RuntimeError

def install(dest_dir, prefix=None, dry_run=False):
    log.info('SLEPc: install')
    if dry_run: return
    if prefix is None:
        prefix = dest_dir
    # Run SLEPc install
    PETSC_ARCH = get_petsc_arch() or ''
    if PETSC_ARCH: PETSC_ARCH = 'PETSC_ARCH=' + PETSC_ARCH
    status = os.system(" ".join((
            find_executable('make'),
            'PETSC_DIR='+get_petsc_dir(), PETSC_ARCH,
            'SLEPC_DESTDIR='+dest_dir,
            'install',
            )))
    if status != 0: raise RuntimeError
    slepcvariables = os.path.join(dest_dir, 'conf', 'slepcvariables')
    fh = open(slepcvariables, 'a')
    fh.write('SLEPC_DESTDIR=%s\n' % prefix)
    fh.close()

class context:
    def __init__(self):
        self.sys_argv = sys.argv[:]
        self.wdir = os.getcwd()
    def enter(self):
        del sys.argv[1:]
        pdir = os.environ['SLEPC_DIR']
        os.chdir(pdir)
        return self
    def exit(self):
        sys.argv[:] = self.sys_argv
        os.chdir(self.wdir)

class cmd_build(_build):

    def initialize_options(self):
        _build.initialize_options(self)
        PETSC_ARCH = os.environ.get('PETSC_ARCH', '')
        self.build_base = os.path.join(PETSC_ARCH, 'build-python')

    def run(self):
        _build.run(self)
        ctx = context().enter()
        try:
            config(self.dry_run)
            build(self.dry_run)
        finally:
            ctx.exit()

class cmd_install(_install):

    def initialize_options(self):
        _install.initialize_options(self)
        self.optimize = 1

    def run(self):
        root_dir = self.install_platlib
        dest_dir = os.path.join(root_dir, 'slepc')
        bdist_base = self.get_finalized_command('bdist').bdist_base
        if dest_dir.startswith(bdist_base):
            prefix = dest_dir[len(bdist_base)+1:]
            prefix = prefix[prefix.index(os.path.sep):]
        else:
            prefix = dest_dir
        dest_dir = os.path.abspath(dest_dir)
        prefix   = os.path.abspath(prefix)
        #
        _install.run(self)
        ctx = context().enter()
        try:
            install(dest_dir, prefix, self.dry_run)
        finally:
            ctx.exit()

manifest_in = """\
include makefile gmakefile
recursive-include config *.py

recursive-include share/slepc/matlab *
recursive-include conf *
recursive-include include *
recursive-include src *

exclude conf/slepcvariables
recursive-exclude src *.html
recursive-exclude src/docs *
recursive-exclude src/*/examples/* *.*
recursive-exclude pypi *
"""

class cmd_sdist(_sdist):

    def initialize_options(self):
        _sdist.initialize_options(self)
        self.force_manifest = 1
        self.template = os.path.join('pypi', 'manifest.in')
        # Generate manifest.in file
        SLEPC_DIR = os.environ['SLEPC_DIR']
        from distutils.dir_util import mkpath
        pkgdir = os.path.join(SLEPC_DIR, 'pypi')
        if not os.path.exists(pkgdir): mkpath(pkgdir)
        template = self.template
        fh = open(template, 'wt')
        fh.write(manifest_in)
        fh.close()

def version():
    import re
    version_re = {
        'major'  : re.compile(r"#define\s+SLEPC_VERSION_MAJOR\s+(\d+)"),
        'minor'  : re.compile(r"#define\s+SLEPC_VERSION_MINOR\s+(\d+)"),
        'micro'  : re.compile(r"#define\s+SLEPC_VERSION_SUBMINOR\s+(\d+)"),
        'patch'  : re.compile(r"#define\s+SLEPC_VERSION_PATCH\s+(\d+)"),
        'release': re.compile(r"#define\s+SLEPC_VERSION_RELEASE\s+(\d+)"),
        }
    slepcversion_h = os.path.join('include','slepcversion.h')
    data = open(slepcversion_h, 'rt').read()
    major = int(version_re['major'].search(data).groups()[0])
    minor = int(version_re['minor'].search(data).groups()[0])
    micro = int(version_re['micro'].search(data).groups()[0])
    patch = int(version_re['patch'].search(data).groups()[0])
    release = int(version_re['release'].search(data).groups()[0])
    if release:
        v = "%d.%d" % (major, minor)
        if micro > 0:
            v += ".%d" % micro
        #if patch > 0:
        #    v += ".post%d" % patch
    else:
        v = "%d.%d.dev%d" % (major, minor+1, 0)
    return v

def tarball():
    VERSION = version()
    if '.dev' in VERSION:
        return None
    bits = VERSION.split('.')
    if len(bits) == 2: bits.append('0')
    SLEPC_VERSION = '.'.join(bits[:3])
    return ('http://slepc.upv.es/download/distrib/'
            'slepc-%s.tar.gz#egg=slepc-%s' % (SLEPC_VERSION, VERSION))

description = __doc__.split('\n')[1:-1]; del description[1:3]
classifiers = """
License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)
Operating System :: POSIX
Intended Audience :: Developers
Intended Audience :: Science/Research
Programming Language :: C
Programming Language :: C++
Programming Language :: Fortran
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries
"""

bootstrap()
setup(name='slepc',
      version=version(),
      description=description.pop(0),
      long_description='\n'.join(description),
      classifiers= classifiers.split('\n')[1:-1],
      keywords = ['SLEPc','PETSc', 'MPI'],
      platforms=['POSIX'],
      license='LGPL',

      url='http://slepc.upv.es/',
      download_url=tarball(),

      author='SLEPc Team',
      author_email='slepc-maint@upv.es',
      maintainer='Lisandro Dalcin',
      maintainer_email='dalcinl@gmail.com',

      packages = ['slepc'],
      package_dir = {'slepc': 'pypi'},
      cmdclass={
        'build': cmd_build,
        'install': cmd_install,
        'sdist': cmd_sdist,
        },
      **metadata)
