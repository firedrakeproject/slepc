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
LOCDIR     = src/mfn/examples/tutorials/
EXAMPLESC  = ex23.c ex26.c
EXAMPLESF  =
MANSEC     = MFN

TESTEXAMPLES_C = ex23.PETSc runex23_1 ex23.rm ex26.PETSc runex26_1 ex26.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex23: ex23.o chkopts
	-${CLINKER} -o ex23 ex23.o ${SLEPC_MFN_LIB}
	${RM} ex23.o

ex26: ex26.o chkopts
	-${CLINKER} -o ex26 ex26.o ${SLEPC_MFN_LIB}
	${RM} ex26.o

#------------------------------------------------------------------------------------

runex23_1:
	-@${MPIEXEC} -n 1 ./ex23 > ex23_1.tmp 2>&1; \
	   if (${DIFF} output/ex23_1.out ex23_1.tmp) then true; \
	   else echo "Possible problem with ex23_1, diffs above"; fi; \
	   ${RM} -f ex23_1.tmp

runex26_1:
	-@${MPIEXEC} -n 1 ./ex26 > ex26_1.tmp 2>&1; \
	   if (${DIFF} output/ex26_1.out ex26_1.tmp) then true; \
	   else echo "Possible problem with ex26_1, diffs above"; fi; \
	   ${RM} -f ex26_1.tmp

