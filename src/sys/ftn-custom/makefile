#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     SLEPc - Scalable Library for Eigenvalue Problem Computations
#     Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain
#
#     This file is part of SLEPc. See the README file for conditions of use
#     and additional information.
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

#requirespackage   'PETSC_HAVE_FORTRAN'
ALL: lib
CFLAGS   = -DSLEPC_LIB_DIR='"${SLEPC_INSTALL_DIR}/lib/${PETSC_ARCH}"'
FFLAGS   =
SOURCEC  = zslepc_start.c zslepc_startf.c
OBJSC    = zslepc_start.o zslepc_startf.o
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libslepc
LOCDIR   = src/sys/ftn-custom/
include ${SLEPC_DIR}/bmake/slepc_common  
