
#requirespackage   'PETSC_HAVE_FORTRAN'
ALL: lib
CFLAGS   = -DSLEPC_LIB_DIR='"${SLEPC_LIB_DIR}"'
FFLAGS   =
SOURCEC  = zslepc_start.c zslepc_startf.c
OBJSC    = zslepc_start.o zslepc_startf.o
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libslepcfortran
LOCDIR   = src/sys/ftn-custom/
include ${SLEPC_DIR}/bmake/slepc_common  
