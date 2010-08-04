#if !defined(__SLEPCVERSION_H)
#define __SLEPCVERSION_H

#define SLEPC_VERSION_RELEASE    0
#define SLEPC_VERSION_MAJOR      3
#define SLEPC_VERSION_MINOR      1
#define SLEPC_VERSION_SUBMINOR   0
#define SLEPC_VERSION_PATCH      0
#define SLEPC_VERSION_DATE       "August 4, 2010"
#define SLEPC_VERSION_PATCH_DATE "unknown"
#define SLEPC_AUTHOR_INFO        "        The SLEPc Team\n\
   slepc-maint@grycap.upv.es\n\
 http://www.grycap.upv.es/slepc\n"

#define SLEPC_VERSION_(MAJOR,MINOR,SUBMINOR) \
 ((SLEPC_VERSION_MAJOR == (MAJOR)) &&       \
  (SLEPC_VERSION_MINOR == (MINOR)) &&       \
  (SLEPC_VERSION_SUBMINOR == (SUBMINOR)) && \
  (SLEPC_VERSION_RELEASE  == 1))

#endif

