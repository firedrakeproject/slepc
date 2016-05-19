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
LOCDIR     = src/sys/classes/st/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c
EXAMPLESF  =
MANSEC     = ST
TESTS      = test1 test2 test3 test4

TESTEXAMPLES_C       = test1.PETSc runtest1_1 runtest1_2 test1.rm \
                       test2.PETSc runtest2_1 runtest2_2 runtest2_3 test2.rm \
                       test3.PETSc runtest3_1 runtest3_2 runtest3_3 test3.rm \
                       test4.PETSc runtest4_1 runtest4_2 runtest4_3 runtest4_4 test4.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_SYS_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_SYS_LIB}
	${RM} test2.o

test3: test3.o chkopts
	-${CLINKER} -o test3 test3.o ${SLEPC_SYS_LIB}
	${RM} test3.o

test4: test4.o chkopts
	-${CLINKER} -o test4 test4.o ${SLEPC_SYS_LIB}
	${RM} test4.o

#------------------------------------------------------------------------------------

TESTCODE = \
	[ x${SAVE_OUTPUT} = xyes ] && cp $${test}.tmp output/$${test}.out; \
	${DIFF} output/$${test}.out $${test}.tmp || \
	echo "Possible problem with $${test}, diffs above"; \
	${RM} -f $${test}.tmp

runtest1_1:
	-@test=test1_1; \
	${MPIEXEC} -n 1 ./test1 -st_matmode inplace 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest1_2:
	-@test=test1_1; \
	${MPIEXEC} -n 1 ./test1 -st_matmode shell 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_1:
	-@test=test2_1; \
	${MPIEXEC} -n 1 ./test2 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_2:
	-@test=test2_1; \
	${MPIEXEC} -n 1 ./test2 -st_matmode inplace 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_3:
	-@test=test2_1; \
	${MPIEXEC} -n 1 ./test2 -st_matmode shell 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest3_1:
	-@test=test3_1; \
	${MPIEXEC} -n 1 ./test3 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest3_2:
	-@test=test3_1; \
	${MPIEXEC} -n 1 ./test3 -st_matmode inplace 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest3_3:
	-@test=test3_1; \
	${MPIEXEC} -n 1 ./test3 -st_matmode shell 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest4_1:
	-@test=test4_1; \
	${MPIEXEC} -n 1 ./test4 -st_transform 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest4_2:
	-@test=test4_1; \
	${MPIEXEC} -n 1 ./test4 -st_transform -st_matmode shell 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest4_3:
	-@test=test4_2; \
	${MPIEXEC} -n 1 ./test4 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest4_4:
	-@test=test4_2; \
	${MPIEXEC} -n 1 ./test4 -st_matmode shell 2>&1 > $${test}.tmp; \
	${TESTCODE}

