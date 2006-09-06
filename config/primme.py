#!/usr/bin/env python

import os
import sys

import petscconf
import log
import check

def Check(conf,directory):
  
  if petscconf.PRECISION == 'single':
    sys.exit('ERROR: PRIMME does not support single precision.')
 
  if not directory:
    sys.exit('ERROR: I cannot guess where is PRIMME directory.')

  functions = ['primme_set_method','primme_Free','primme_initialize']
  if petscconf.SCALAR == 'real':
    functions += ['dprimme']
    include = 'DPRIMME'
    lib = '-ldprimme'
  else:
    functions += ['zprimme']
    include = 'ZPRIMME'
    lib = '-lzprimme'
    
  
  flags = ['-I' + directory + '/' + include]
  libs =  ['-L' + directory + ' ' + lib]
  
  if check.Link(functions,[],flags+libs):
    conf.write('SLEPC_HAVE_PRIMME = -DSLEPC_HAVE_PRIMME\n')
    conf.write('PRIMME_LIB =' + str.join(' ', libs) + '\n')
    conf.write('PRIMME_FLAGS =' + str.join(' ', flags) + '\n')
  else:
    sys.exit('ERROR: PRIMME link test failed.')
  
  return flags + libs
 