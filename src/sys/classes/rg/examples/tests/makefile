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
LOCDIR     = src/sys/classes/rg/examples/tests/
EXAMPLESC  = test1.c test2.c
EXAMPLESF  = test1f.F
MANSEC     = RG
TESTS      = test1 test1f test2

TESTEXAMPLES_C       = test1.PETSc runtest1_1 test1.rm \
                       test2.PETSc runtest2_1 runtest2_2 test2.rm
TESTEXAMPLES_FORTRAN = test1f.PETSc runtest1f_1 test1f.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_SYS_LIB}
	${RM} test1.o

test1f: test1f.o chkopts
	-${FLINKER} -o test1f test1f.o ${SLEPC_SYS_LIB}
	${RM} test1f.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_SYS_LIB}
	${RM} test2.o

#------------------------------------------------------------------------------------

runtest1_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test1 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test1f > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test2 -rg_ring_width 0.015 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test2 -rg_ring_width 0.015 -rg_scale 1.5 > $${test}.tmp 2>&1; \
	${TESTCODE}

