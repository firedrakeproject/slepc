#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     SLEPc - Scalable Library for Eigenvalue Problem Computations
#     Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain
#
#     This file is part of SLEPc. See the README file for conditions of use
#     and additional information.
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

ALL: lib

#requirespackage  'PETSC_HAVE_BLOPEX'
#requiresprecision double
#requiresscalar    real

CFLAGS   = ${BLOPEX_INCLUDE}
FFLAGS   =
SOURCEC  = blopex.c slepc-interface.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = blopex.o slepc-interface.o
LIBBASE  = libslepc
DIRS     =
MANSEC   = EPS
LOCDIR   = src/eps/impls/blopex/

include ${SLEPC_DIR}/conf/slepc_common


