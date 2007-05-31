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

CFLAGS   = -DSLEPC_LIB_DIR='"${SLEPC_INSTALL_DIR}/lib/${PETSC_ARCH}"'
FFLAGS   =
SOURCEC  = slepcinit.c slepcutil.c
SOURCEF  =
SOURCEH  = 
OBJSC    = slepcinit.o slepcutil.o
OBJSF    = 
LIBBASE  = libslepc
MANSEC   = sys
LOCDIR   = src/sys/

include ${SLEPC_DIR}/bmake/slepc_common

