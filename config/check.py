import os
import sys
import commands

import petscconf
import log
 
def Link(functions,callbacks,flags):
  os.chdir('config')
  cfile = open('checklink.c','w')
  cfile.write('#include "petsc.h"\n')

  cfile.write('EXTERN_C_BEGIN\n')
  for f in functions:
    cfile.write('EXTERN int\n')
    cfile.write(f)
    cfile.write('();\n')
  cfile.write('EXTERN_C_END\n')
  
  for c in callbacks:
    cfile.write('int ')    
    cfile.write(c)
    cfile.write('() { return 0; } \n')    

  cfile.write('int main() {\n')
  for f in functions:
    cfile.write(f)
    cfile.write('();\n')
  cfile.write('return 0;\n}\n')
  cfile.close()
  (result, output) = commands.getstatusoutput(petscconf.MAKE + ' checklink TESTFLAGS="'+str.join(' ',flags)+'"')
  os.chdir(os.pardir)
  if result:
    log.Write(output)
    return 0
  else:
    return 1

def FortranLink(functions,callbacks,flags):
  log.Write('=== With linker flags: '+str.join(' ',flags))
  f = []
  for i in functions:
    f.append(i+'_')
  c = []
  for i in callbacks:
    c.append(i+'_')  
  if Link(f,c,flags): return 'UNDERSCORE'
  f = []
  for i in functions:
    f.append(i.upper())
  c = []
  for i in callbacks:
    c.append(i.upper())  
  if Link(f,c,flags): return 'CAPS'
  if Link(functions,callbacks,flags): return 'STDCALL'
  return ''

def GenerateGuesses(name):
  installdirs = ['/usr/local','/opt']
  if 'HOME' in os.environ:
    installdirs.insert(0,os.environ['HOME'])

  dirs = []
  for i in installdirs:
    dirs = dirs + [i + '/lib']
    for d in [name,name.upper(),name.lower()]:
      dirs = dirs + [i + '/' + d]
      dirs = dirs + [i + '/' + d + '/lib']
      dirs = dirs + [i + '/lib/' + d]
      
  for d in dirs[:]:
    if not os.path.exists(d):
      dirs.remove(d)
  dirs = [''] + dirs
  return dirs

def FortranLib(conf,name,dirs,libs,functions,callbacks = []):
  log.Println('Checking '+name+' library...')

  mangling = 0
  for d in dirs:
    for l in libs:
      if d:
	flags = ['-L' + d] + l
      else:
	flags = l
      mangling = FortranLink(functions,callbacks,flags)
      if mangling: break
    if mangling: break    

  if not mangling:
    print 'ERROR: Unable to link with library',name
    print 'ERROR: In directories',dirs
    print 'ERROR: With flags',libs
    sys.exit(1)
    

  conf.write('SLEPC_HAVE_' + name + ' = -DSLEPC_HAVE_' + name + ' -DSLEPC_' + name + '_HAVE_'+mangling+'\n')
  conf.write(name + '_LIB = '+str.join(' ',flags)+'\n')
  return flags