#if !defined(__SLEPCVERSION_H)
#define __SLEPCVERSION_H

#define SLEPC_VERSION_RELEASE    0
#define SLEPC_VERSION_MAJOR      3
#define SLEPC_VERSION_MINOR      4
#define SLEPC_VERSION_SUBMINOR   0
#define SLEPC_VERSION_PATCH      0
#define SLEPC_VERSION_DATE       "July 5, 2013"
#define SLEPC_VERSION_PATCH_DATE "unknown"

#if !defined (SLEPC_VERSION_SVN)
#define SLEPC_VERSION_SVN        "unknown"
#endif

#if !defined(SLEPC_VERSION_DATE_SVN)
#define SLEPC_VERSION_DATE_SVN   "unknown"
#endif

#define SLEPC_VERSION_(MAJOR,MINOR,SUBMINOR) \
 ((SLEPC_VERSION_MAJOR == (MAJOR)) &&       \
  (SLEPC_VERSION_MINOR == (MINOR)) &&       \
  (SLEPC_VERSION_SUBMINOR == (SUBMINOR)) && \
  (SLEPC_VERSION_RELEASE  == 1))

#endif

