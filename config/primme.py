#!/usr/bin/env python

import os
import sys

import petscconf
import log
import check

def Check(conf,directory,libs):
  
  if petscconf.PRECISION == 'single':
    sys.exit('ERROR: PRIMME does not support single precision.')
 
  functions = ['primme_set_method','primme_Free','primme_initialize']
  if petscconf.SCALAR == 'real':
    functions += ['dprimme']
    include = 'DPRIMME'
    lib = str.join(' ', libs) + ' -ldprimme'
  else:
    functions += ['zprimme']
    include = 'ZPRIMME'
    lib = str.join(' ', libs) + ' -lzprimme'
    
  if directory:
    directory = [directory]
  else:
    directory = check.GenerateGuesses('PRIMME')

  for dir in directory:
    flags = ['-I' + dir + '/' + include]
    libs =  ['-L' + dir + ' ' + lib]
    if check.Link(functions,[],flags+libs):
      conf.write('SLEPC_HAVE_PRIMME = -DSLEPC_HAVE_PRIMME\n')
      conf.write('PRIMME_LIB =' + str.join(' ', libs) + '\n')
      conf.write('PRIMME_FLAGS =' + str.join(' ', flags) + '\n')
      return flags + libs 

  sys.exit('ERROR: PRIMME link test failed.')

