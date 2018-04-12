
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
#requiresdefine   'PETSC_HAVE_FORTRAN'
#requiresdefine   'PETSC_USING_F90'
#
#   Makes Fortran module functions from *.h90 files in include/slepc/finclude
#
ALL: buildmod_slepc
speciallib: buildmod_slepc
specialfastlib: buildmod_slepc
SPECIALLIB     = yes
SPECIALFASTLIB = yes


CFLAGS     =
FFLAGS     =
SOURCEC    =
SOURCEF    = slepcmfnmod.F
SOURCEH    = slepcmfn.h slepcmfn.h90
LIBBASE    = libslepcmfn
MANSEC     = MFN
LOCDIR     = src/mfn/f90-mod/
CLEANFILES = *.mod

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

