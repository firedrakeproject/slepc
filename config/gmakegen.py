#!/usr/bin/env python

import os
from distutils.sysconfig import parse_makefile
import sys
import logging
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config'))
from cmakegen import Mistakes, stripsplit, AUTODIRS, SKIPDIRS
from cmakegen import defaultdict # collections.defaultdict, with fallback for python-2.4

PKGS = 'sys eps svd pep nep mfn lme'.split()
LANGS = dict(c='C', cxx='CXX', cu='CU', F='F', F90='F90')

try:
    all([True, True])
except NameError:               # needs python-2.5
    def all(iterable):
        for i in iterable:
            if not i:
                return False
        return True

try:
    os.path.relpath             # needs python-2.6
except AttributeError:
    def _relpath(path, start=os.path.curdir):
        """Return a relative version of a path"""

        from os.path import curdir, abspath, commonprefix, sep, pardir, join
        if not path:
            raise ValueError("no path specified")

        start_list = [x for x in abspath(start).split(sep) if x]
        path_list = [x for x in abspath(path).split(sep) if x]

        # Work out how much of the filepath is shared by start and path.
        i = len(commonprefix([start_list, path_list]))

        rel_list = [pardir] * (len(start_list)-i) + path_list[i:]
        if not rel_list:
            return curdir
        return join(*rel_list)
    os.path.relpath = _relpath

class debuglogger(object):
    def __init__(self, log):
        self._log = log

    def write(self, string):
        self._log.debug(string)

class Slepc(object):
    def __init__(self, slepc_dir=None, petsc_dir=None, petsc_arch=None, installed_petsc=False, verbose=False):
        if slepc_dir is None:
            slepc_dir = os.environ.get('SLEPC_DIR')
            if slepc_dir is None:
                raise RuntimeError('Could not determine SLEPC_DIR, please set in environment')
        if petsc_dir is None:
            petsc_dir = os.environ.get('PETSC_DIR')
            if petsc_dir is None:
                raise RuntimeError('Could not determine PETSC_DIR, please set in environment')
        if petsc_arch is None:
            petsc_arch = os.environ.get('PETSC_ARCH')
            if petsc_arch is None:
                raise RuntimeError('Could not determine PETSC_ARCH, please set in environment')
        self.slepc_dir = slepc_dir
        self.petsc_dir = petsc_dir
        self.petsc_arch = petsc_arch
        self.installed_petsc = installed_petsc
        self.read_conf()
        logging.basicConfig(filename=self.arch_path('lib','slepc','conf', 'gmake.log'), level=logging.DEBUG)
        self.log = logging.getLogger('gmakegen')
        self.mistakes = Mistakes(debuglogger(self.log), verbose=verbose)
        self.gendeps = []

    def petsc_path(self, *args):
        if self.installed_petsc:
            return os.path.join(self.petsc_dir, *args)
        else:
            return os.path.join(self.petsc_dir, self.petsc_arch, *args)

    def arch_path(self, *args):
        return os.path.join(self.slepc_dir, self.petsc_arch, *args)

    def read_conf(self):
        self.conf = dict()
        for line in open(self.petsc_path('include', 'petscconf.h')):
            if line.startswith('#define '):
                define = line[len('#define '):]
                space = define.find(' ')
                key = define[:space]
                val = define[space+1:]
                self.conf[key] = val
        self.conf.update(parse_makefile(self.petsc_path('lib','petsc','conf', 'petscvariables')))
        for line in open(self.arch_path('include', 'slepcconf.h')):
            if line.startswith('#define '):
                define = line[len('#define '):]
                space = define.find(' ')
                key = define[:space]
                val = define[space+1:]
                self.conf[key] = val
        self.conf.update(parse_makefile(self.arch_path('lib','slepc','conf', 'slepcvariables')))
        self.have_fortran = int(self.conf.get('PETSC_HAVE_FORTRAN', '0'))

    def inconf(self, key, val):
        if key in ['package', 'function', 'define']:
            return self.conf.get(val)
        elif key == 'precision':
            return val == self.conf['PETSC_PRECISION']
        elif key == 'scalar':
            return val == self.conf['PETSC_SCALAR']
        elif key == 'language':
            return val == self.conf['PETSC_LANGUAGE']
        raise RuntimeError('Unknown conf check: %s %s' % (key, val))

    def relpath(self, root, src):
        return os.path.relpath(os.path.join(root, src), self.slepc_dir)

    def get_sources(self, makevars):
        """Return dict {lang: list_of_source_files}"""
        source = dict()
        for lang, sourcelang in LANGS.items():
            source[lang] = [f for f in makevars.get('SOURCE'+sourcelang,'').split() if f.endswith(lang)]
        return source

    def gen_pkg(self, pkg):
        pkgsrcs = dict()
        for lang in LANGS:
            pkgsrcs[lang] = []
        for root, dirs, files in os.walk(os.path.join(self.slepc_dir, 'src', pkg)):
            makefile = os.path.join(root,'makefile')
            if not os.path.exists(makefile):
                dirs[:] = []
                continue
            mklines = open(makefile)
            conditions = set(tuple(stripsplit(line)) for line in mklines if line.startswith('#requires'))
            mklines.close()
            if not all(self.inconf(key, val) for key, val in conditions):
                dirs[:] = []
                continue
            makevars = parse_makefile(makefile)
            mdirs = makevars.get('DIRS','').split() # Directories specified in the makefile
            self.mistakes.compareDirLists(root, mdirs, dirs) # diagnostic output to find unused directories
            candidates = set(mdirs).union(AUTODIRS).difference(SKIPDIRS)
            dirs[:] = list(candidates.intersection(dirs))
            allsource = []
            def mkrel(src):
                return self.relpath(root, src)
            source = self.get_sources(makevars)
            for lang, s in source.items():
                pkgsrcs[lang] += map(mkrel, s)
                allsource += s
            self.mistakes.compareSourceLists(root, allsource, files) # Diagnostic output about unused source files
            self.gendeps.append(self.relpath(root, 'makefile'))
        return pkgsrcs

    def gen_gnumake(self, fd):
        def write(stem, srcs):
            fd.write('%s :=\n' % stem)
            for lang in LANGS:
                fd.write('%(stem)s.%(lang)s := %(srcs)s\n' % dict(stem=stem, lang=lang, srcs=' '.join(srcs[lang])))
                fd.write('%(stem)s += $(%(stem)s.%(lang)s)\n' % dict(stem=stem, lang=lang))
        for pkg in PKGS:
            srcs = self.gen_pkg(pkg)
            write('srcs-' + pkg, srcs)
        return self.gendeps

    def summary(self):
        self.mistakes.summary()

def WriteGnuMake(slepc):
    arch_files = slepc.arch_path('lib','slepc','conf', 'files')
    fd = open(arch_files, 'w')
    gendeps = slepc.gen_gnumake(fd)
    fd.write('\n')
    fd.write('# Dependency to regenerate this file\n')
    fd.write('%s : %s %s\n' % (os.path.relpath(arch_files, slepc.slepc_dir),
                               os.path.relpath(__file__, os.path.realpath(slepc.slepc_dir)),
                               ' '.join(gendeps)))
    fd.write('\n')
    fd.write('# Dummy dependencies in case makefiles are removed\n')
    fd.write(''.join([dep + ':\n' for dep in gendeps]))
    fd.close()

def main(slepc_dir=None, petsc_dir=None, petsc_arch=None, installed_petsc=False, output=None, verbose=False):
    if output is None:
        output = 'gnumake'
    writer = dict(gnumake=WriteGnuMake)
    slepc = Slepc(slepc_dir=slepc_dir, petsc_dir=petsc_dir, petsc_arch=petsc_arch, installed_petsc=installed_petsc, verbose=verbose)
    writer[output](slepc)
    slepc.summary()

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--verbose', help='Show mismatches between makefiles and the filesystem', action='store_true', default=False)
    parser.add_option('--petsc-arch', help='Set PETSC_ARCH different from environment', default=os.environ.get('PETSC_ARCH'))
    parser.add_option('--installed-petsc', help='Using a prefix PETSc installation', default=False)
    parser.add_option('--output', help='Location to write output file', default=None)
    opts, extra_args = parser.parse_args()
    if extra_args:
        import sys
        sys.stderr.write('Unknown arguments: %s\n' % ' '.join(extra_args))
        exit(1)
    main(petsc_arch=opts.petsc_arch, installed_petsc=opts.installed_petsc, output=opts.output, verbose=opts.verbose)
