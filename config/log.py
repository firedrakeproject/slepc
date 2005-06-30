import sys

def Open(filename):
  global f
  f = open(filename,'w')
  return

def Println(string):
  print string
  f.write(string)
  f.write('\n')

def Print(string):
  print string,
  f.write(string+' ')
  
def Write(string):
  f.write(string)
  f.write('\n')
  
def Exit(string):
  Println(string)
  f.close()
  sys.exit(string)

