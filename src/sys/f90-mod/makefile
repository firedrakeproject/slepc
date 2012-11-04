#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain
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
#   Makes Fortran module functions from *.h90 files in include/finclude
#
ALL: buildmod_slepc
speciallib: buildmod_slepc
specialfastlib: buildmod_slepc
SPECIALLIB     = yes
SPECIALFASTLIB = yes


CFLAGS     = 
FFLAGS     = 
SOURCEC    = 
SOURCEF    = slepcmod.F
SOURCEH    = 
LIBBASE    = libslepc
MANSEC     = sys
LOCDIR     = src/sys/f90-mod/
CLEANFILES = *.mod

include ${SLEPC_DIR}/conf/slepc_common

buildmod_slepc:
	-@${OMAKE}  clean
	@${OMAKE}  libfastcompile
	@${OMAKE}  modcopy_slepc
	@${OMAKE}  clean
modcopy_slepc:
	@${CP} -f *.mod ${SLEPC_DIR}/${PETSC_ARCH}/include
