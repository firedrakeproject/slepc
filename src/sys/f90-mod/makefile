
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#
#  SLEPc is free software: you can redistribute it and/or modify it under  the
#  terms of version 3 of the GNU Lesser General Public License as published by
#  the Free Software Foundation.
#
#  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
#  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
#  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
#  more details.
#
#  You  should have received a copy of the GNU Lesser General  Public  License
#  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
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
SOURCEF    = slepcsysmod.F
SOURCEH    =
LIBBASE    = libslepcsys
MANSEC     = sys
LOCDIR     = src/sys/f90-mod/
CLEANFILES = *.mod

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

