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
LOCDIR     = src/nep/examples/tests/
EXAMPLESC  = test1.c test2.c
EXAMPLESF  = test2f.F
MANSEC     = NEP
TESTS      = test1 test2 test2f

TESTEXAMPLES_C_NOTSINGLE       = test1.PETSc runtest1_1 test1.rm \
                                 test2.PETSc runtest2_1 test2.rm
TESTEXAMPLES_FORTRAN_NOTSINGLE = test2f.PETSc runtest2f_1 test2f.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_NEP_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_NEP_LIB}
	${RM} test2.o

test2f: test2f.o chkopts
	-${FLINKER} -o test2f test2f.o ${SLEPC_NEP_LIB}
	${RM} test2f.o

#------------------------------------------------------------------------------------

runtest1_1: runtest1_1_rii runtest1_1_slp
runtest1_1_%:
	-@${SETTEST}; check=test1_1; nep=$*; \
	${MPIEXEC} -n 1 ./test1 -nep_type $${nep} -nep_target 21 -terse >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test2 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test2f > $${test}.tmp 2>&1; \
	${TESTCODE}

