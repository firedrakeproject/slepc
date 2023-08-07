#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
try:
  import petsclinter as pl
except ModuleNotFoundError as mnfe:
  try:
    petsc_dir = os.environ['PETSC_DIR']
  except KeyError as ke:
    raise RuntimeError('Must set PETSC_DIR environment variable') from ke
  sys.path.insert(0, os.path.join(petsc_dir, 'lib', 'petsc', 'bin', 'maint', 'petsclinter'))
  import petsclinter as pl

def __prepare_ns_args(ns_args, parser):
  slepc_mansecs     = ['eps','lme','mfn','nep','pep','svd','sys']
  slepc_aux_mansecs = ['bv','ds','fn','rg','st']

  if ns_args.slepc_dir is None:
    raise RuntimeError('Could not determine SLEPC_DIR from environment, please set via options')

  extra_compiler_flags = [
    '-I' + os.path.join(ns_args.slepc_dir, 'include'),
    '-I' + os.path.join(ns_args.slepc_dir, ns_args.petsc_arch, 'include')
  ]
  with open(os.path.join(ns_args.slepc_dir, ns_args.petsc_arch, 'lib', 'slepc', 'conf', 'slepcvariables'), 'r') as sv:
    line = sv.readline()
    while line:
      if 'INCLUDE' in line:
        for inc in line.split('=', 1)[1].split():
          extra_compiler_flags.append(inc)
      line = sv.readline()

  extra_header_includes = []
  mansecimpls           = [m + 'impl.h' for m in slepc_mansecs + slepc_aux_mansecs] + [
    'slepcimpl.h', 'vecimplslepc.h'
  ]
  for header_file in os.listdir(os.path.join(ns_args.slepc_dir, 'include', 'slepc', 'private')):
    if header_file in mansecimpls:
      extra_header_includes.append(f'#include <slepc/private/{header_file}>')

  if ns_args.src_path == parser.get_default('src_path'):
    ns_args.src_path = os.path.join(ns_args.slepc_dir, 'src')
  if ns_args.patch_dir == parser.get_default('patch_dir'):
    ns_args.patch_dir = os.path.join(ns_args.slepc_dir, 'slepcLintPatches')

  # prepend these
  ns_args.extra_compiler_flags = extra_compiler_flags + ns_args.extra_compiler_flags
  # replace these
  if not ns_args.extra_header_includes:
    ns_args.extra_header_includes = extra_header_includes

  return ns_args

def command_line_main():
  import argparse
  import petsclinter.main

  slepc_classid_map = {
    '_p_BV *'  : 'BV_CLASSID',
    '_p_DS *'  : 'DS_CLASSID',
    '_p_FN *'  : 'FN_CLASSID',
    '_p_RG *'  : 'RG_CLASSID',
    '_p_ST *'  : 'ST_CLASSID',
    '_p_EPS *' : 'EPS_CLASSID',
    '_p_PEP *' : 'PEP_CLASSID',
    '_p_NEP *' : 'NEP_CLASSID',
    '_p_SVD *' : 'SVD_CLASSID',
    '_p_MFN *' : 'MFN_CLASSID',
    '_p_LME *' : 'LME_CLASSID',
  }

  for struct_name, classid_name in slepc_classid_map.items():
    pl.checks.register_classid(struct_name, classid_name)

  parser      = argparse.ArgumentParser(prog='slepclinter', add_help=False)
  group_slepc = parser.add_argument_group(title='SLEPc location settings')
  group_slepc.add_argument('--SLEPC_DIR', required=False, default=os.environ.get('SLEPC_DIR', None), help='if this option is unused defaults to environment variable $SLEPC_DIR', dest='slepc_dir')

  args, parser = pl.main.parse_command_line_args(parent_parsers=[parser])
  args         = __prepare_ns_args(args, parser)

  return pl.main.namespace_main(args)

if __name__ == '__main__':
  sys.exit(command_line_main())
