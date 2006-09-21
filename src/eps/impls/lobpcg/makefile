
ALL: lib

#requirespackage  'PETSC_HAVE_HYPRE'
#requiresprecision double
#requiresscalar    real
#requireslanguage CONLY

CFLAGS   = ${HYPRE_INCLUDE}
FFLAGS   =
SOURCEC  = lobpcg.c petsc-interface.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = lobpcg.o petsc-interface.o
LIBBASE  = libslepc
DIRS     =
MANSEC   = EPS
LOCDIR   = src/eps/impls/lobpcg/

include ${SLEPC_DIR}/bmake/slepc_common


