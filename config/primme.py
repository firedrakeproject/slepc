#!/usr/bin/env python

import os
import sys

import petscconf
import log
import check

def Check(conf,directory,libs):
  
  log.Write('='*80)
  log.Println('Checking PRIMME library...')

  if petscconf.PRECISION == 'single':
    sys.exit('ERROR: PRIMME does not support single precision.')
 
  functions_base = ['primme_set_method','primme_Free','primme_initialize']
  if directory:
    dirs = [directory]
  else:
    dirs = check.GenerateGuesses('Primme')

  include = 'PRIMMESRC/COMMONSRC'
  if not libs:
    libs = ['-lprimme']
  if petscconf.SCALAR == 'real':
    functions = functions_base + ['dprimme']
  else:
    functions = functions_base + ['zprimme']

  for d in dirs:
    if d:
      l = ['-L' + d] + libs
      f = ['-I' + d + '/' + include]
    else:
      l =  libs
      f = []
    if check.Link(functions,[],l+f):
      conf.write('SLEPC_HAVE_PRIMME = -DSLEPC_HAVE_PRIMME\n')
      conf.write('PRIMME_LIB =' + str.join(' ', l) + '\n')
      conf.write('PRIMME_FLAGS =' + str.join(' ', f) + '\n')
      return l+f

  log.Println('ERROR: Unable to link with PRIMME library')
  print 'ERROR: In directories',dirs
  print 'ERROR: With flags',libs,
  log.Exit('')
