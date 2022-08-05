# --------------------------------------------------------------------

__all__ = ['setup',
           'Extension',
           'config',
           'build',
           'build_src',
           'build_ext',
           'install',
           'clean',
           'test',
           'sdist',
           ]

# --------------------------------------------------------------------

import sys, os

from conf.baseconf import PetscConfig
from conf.baseconf import setup, Extension, log, strip_prefix
from conf.baseconf import config     as _config
from conf.baseconf import build      as _build
from conf.baseconf import build_src  as _build_src
from conf.baseconf import build_ext  as _build_ext
from conf.baseconf import install    as _install
from conf.baseconf import clean      as _clean
from conf.baseconf import test       as _test
from conf.baseconf import sdist      as _sdist
from conf.baseconf import makefile

from distutils.errors import DistutilsError
from distutils.util import split_quoted

# --------------------------------------------------------------------


class SlepcConfig(PetscConfig):

    def __init__(self,  slepc_dir, petsc_dir, petsc_arch, dest_dir=None):
        PetscConfig.__init__(self, petsc_dir, petsc_arch, dest_dir='')
        if dest_dir is None:
            dest_dir = os.environ.get('DESTDIR')
        if not slepc_dir:
            raise DistutilsError("SLEPc not found")
        if not os.path.isdir(slepc_dir):
            raise DistutilsError("invalid SLEPC_DIR")
        self.sversion = self._get_slepc_version(slepc_dir)
        self._get_slepc_config(petsc_dir,slepc_dir)
        self.SLEPC_DIR = self['SLEPC_DIR']
        self.SLEPC_DESTDIR = dest_dir
        self.SLEPC_LIB = self['SLEPC_LIB']
        self.SLEPC_EXTERNAL_LIB_DIR = self['SLEPC_EXTERNAL_LIB_DIR']

    def _get_slepc_version(self, slepc_dir):
        import re
        version_re = {
            'major'  : re.compile(r"#define\s+SLEPC_VERSION_MAJOR\s+(\d+)"),
            'minor'  : re.compile(r"#define\s+SLEPC_VERSION_MINOR\s+(\d+)"),
            'micro'  : re.compile(r"#define\s+SLEPC_VERSION_SUBMINOR\s+(\d+)"),
            'release': re.compile(r"#define\s+SLEPC_VERSION_RELEASE\s+(-*\d+)"),
            }
        slepcversion_h = os.path.join(slepc_dir, 'include', 'slepcversion.h')
        with open(slepcversion_h, 'rt') as f: data = f.read()
        major = int(version_re['major'].search(data).groups()[0])
        minor = int(version_re['minor'].search(data).groups()[0])
        micro = int(version_re['micro'].search(data).groups()[0])
        release = int(version_re['release'].search(data).groups()[0])
        return  (major, minor, micro), (release == 1)

    def _get_slepc_config(self, petsc_dir, slepc_dir):
        from os.path import join, isdir
        PETSC_DIR  = petsc_dir
        SLEPC_DIR  = slepc_dir
        PETSC_ARCH = self.PETSC_ARCH
        confdir = join('lib', 'slepc', 'conf')
        if not (PETSC_ARCH and isdir(join(SLEPC_DIR, PETSC_ARCH))):
            PETSC_ARCH = ''
        variables = join(SLEPC_DIR, confdir, 'slepc_variables')
        slepcvariables = join(SLEPC_DIR, PETSC_ARCH, confdir, 'slepcvariables')
        with open(variables) as f:
            contents = f.read()
        with open(slepcvariables) as f:
            contents += f.read()
        try:
            from cStringIO import StringIO
        except ImportError:
            from io import StringIO
        confstr  = 'PETSC_DIR  = %s\n' % PETSC_DIR
        confstr += 'PETSC_ARCH = %s\n' % PETSC_ARCH
        confstr  = 'SLEPC_DIR  = %s\n' % SLEPC_DIR
        confstr += contents
        slepc_confdict = makefile(StringIO(confstr))
        self.configdict['SLEPC_DIR'] = SLEPC_DIR
        self.configdict['SLEPC_LIB'] = slepc_confdict['SLEPC_LIB']
        dirlist = []
        for external in [ 'ARPACK_LIB', 'BLOPEX_LIB', 'ELPA_LIB', 'EVSL_LIB', 'HPDDM_LIB', 'PRIMME_LIB', 'SLICOT_LIB', 'TRLAN_LIB' ]:
            for entry in [lib[2:] for lib in split_quoted(slepc_confdict[external]) if lib.startswith('-L')]:
                if entry not in dirlist:
                    dirlist.append(entry)
        self.configdict['SLEPC_EXTERNAL_LIB_DIR'] = dirlist

    def configure_extension(self, extension):
        PetscConfig.configure_extension(self, extension)
        SLEPC_DIR  = self.SLEPC_DIR
        PETSC_ARCH = self.PETSC_ARCH
        SLEPC_DESTDIR = self.SLEPC_DESTDIR
        # take into account the case of prefix PETSc with non-prefix SLEPc
        SLEPC_ARCH_DIR = PETSC_ARCH if PETSC_ARCH else os.environ.get('PETSC_ARCH', '')
        # includes and libraries
        SLEPC_INCLUDE = [
            os.path.join(SLEPC_DIR, SLEPC_ARCH_DIR, 'include'),
            os.path.join(SLEPC_DIR, 'include'),
            ]
        SLEPC_LIB_DIR = [
            os.path.join(SLEPC_DIR, SLEPC_ARCH_DIR, 'lib'),
            os.path.join(SLEPC_DIR, 'lib'),
            ] + self.SLEPC_EXTERNAL_LIB_DIR
        slepc_cfg = { }
        slepc_cfg['include_dirs'] = SLEPC_INCLUDE
        slepc_cfg['library_dirs'] = SLEPC_LIB_DIR
        slepc_cfg['libraries'] = [lib[2:] for lib in split_quoted(self.SLEPC_LIB) if lib.startswith('-l')]
        slepc_cfg['runtime_library_dirs'] = [strip_prefix(SLEPC_DESTDIR, d) for d in SLEPC_LIB_DIR]
        self._configure_ext(extension, slepc_cfg, preppend=True)
        if self['BUILDSHAREDLIB'] == 'no':
            from petsc4py.lib import ImportPETSc
            PETSc = ImportPETSc(PETSC_ARCH)
            extension.extra_objects.append(PETSc.__file__)

        # extra configuration
        cflags = []
        extension.extra_compile_args.extend(cflags)
        lflags = []
        extension.extra_link_args.extend(lflags)

    def log_info(self):
        if not self.SLEPC_DIR: return
        version = ".".join([str(i) for i in self.sversion[0]])
        release = ("development", "release")[self.sversion[1]]
        version_info = version + ' ' + release
        log.info('SLEPC_DIR:    %s' % self.SLEPC_DIR)
        log.info('version:      %s' % version_info)
        PetscConfig.log_info(self)


# --------------------------------------------------------------------

cmd_slepc_opts = [
    ('slepc-dir=', None,
     "define SLEPC_DIR, overriding environmental variable.")
    ]

class config(_config):

    Configure = SlepcConfig

    user_options = _config.user_options + cmd_slepc_opts

    def initialize_options(self):
        _config.initialize_options(self)
        self.slepc_dir  = None

    def get_config_arch(self, arch):
        return config.Configure(self.slepc_dir, self.petsc_dir, arch)

    def run(self):
        self.slepc_dir = config.get_slepc_dir(self.slepc_dir)
        if self.slepc_dir is None: return
        log.info('-' * 70)
        log.info('SLEPC_DIR:   %s' % self.slepc_dir)
        _config.run(self)

    #@staticmethod
    def get_slepc_dir(slepc_dir):
        if not slepc_dir: return None
        slepc_dir = os.path.expandvars(slepc_dir)
        if not slepc_dir or '$SLEPC_DIR' in slepc_dir:
            try:
                import slepc
                slepc_dir = slepc.get_slepc_dir()
            except ImportError:
                log.warn("SLEPC_DIR not specified")
                return None
        slepc_dir = os.path.expanduser(slepc_dir)
        slepc_dir = os.path.abspath(slepc_dir)
        if not os.path.isdir(slepc_dir):
            log.warn('invalid SLEPC_DIR:  %s' % slepc_dir)
            return None
        return slepc_dir
    get_slepc_dir = staticmethod(get_slepc_dir)

class build(_build):

    user_options = _build.user_options + cmd_slepc_opts

    def initialize_options(self):
        _build.initialize_options(self)
        self.slepc_dir  = None

    def finalize_options(self):
        _build.finalize_options(self)
        self.set_undefined_options('config',
                                   ('slepc_dir', 'slepc_dir'),)
        self.slepc_dir = config.get_slepc_dir(self.slepc_dir)


class build_src(_build_src):
    pass


class build_ext(_build_ext):

    user_options = _build_ext.user_options + cmd_slepc_opts

    def initialize_options(self):
        _build_ext.initialize_options(self)
        self.slepc_dir  = None

    def finalize_options(self):
        _build_ext.finalize_options(self)
        self.set_undefined_options('build',
                                   ('slepc_dir',  'slepc_dir'))

    def get_config_arch(self, arch):
        return config.Configure(self.slepc_dir, self.petsc_dir, arch)

    def get_config_data(self, arch_list):
        template = """\
SLEPC_DIR  = %(SLEPC_DIR)s
PETSC_DIR  = %(PETSC_DIR)s
PETSC_ARCH = %(PETSC_ARCH)s
"""
        SLEPC_DESTDIR = None
        for arch in arch_list:
            conf = self.get_config_arch(arch)
            SLEPC_DESTDIR = conf.SLEPC_DESTDIR # all archs will have same value
        variables = {'SLEPC_DIR'  : strip_prefix(SLEPC_DESTDIR, self.slepc_dir),
                     'PETSC_DIR'  : self.petsc_dir,
                     'PETSC_ARCH' : os.path.pathsep.join(arch_list)}
        return template, variables


class install(_install):
    pass

class clean(_clean):
    pass

class test(_test):
    pass

class sdist(_sdist):
    pass

# --------------------------------------------------------------------
