#
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

CFLAGS     =
FFLAGS     =
CPPFLAGS   =
FPPFLAGS   =
LOCDIR     = src/lme/examples/tutorials/
EXAMPLESC  = ex32.c
EXAMPLESF  =
MANSEC     = LME

TESTEXAMPLES_C_NOTSINGLE = ex32.PETSc runex32_1 ex32.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex32: ex32.o chkopts
	-${CLINKER} -o ex32 ex32.o ${SLEPC_LME_LIB}
	${RM} ex32.o

#------------------------------------------------------------------------------------

runex32_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex32 > $${test}.tmp 2>&1; \
	${TESTCODE}

