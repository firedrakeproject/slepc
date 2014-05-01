#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain
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
LOCDIR     = src/sys/classes/bv/examples/tests/
EXAMPLESC  = test1.c test2.c
EXAMPLESF  =
MANSEC     = BV
TESTS      = test1 test2

TESTEXAMPLES_C       = test1.PETSc runtest1_1 runtest1_2 runtest1_3 runtest1_4 test1.rm \
                       test2.PETSc runtest2_1 runtest2_2 runtest2_3 runtest2_4 \
                                   runtest2_5 runtest2_6 runtest2_7 runtest2_8 test2.rm

include ${SLEPC_DIR}/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_SYS_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_SYS_LIB}
	${RM} test2.o

#------------------------------------------------------------------------------------

TESTCODE = \
	[ x${SAVE_OUTPUT} = xyes ] && cp $${test}.tmp output/$${test}.out; \
	${DIFF} output/$${test}.out $${test}.tmp || \
	echo "Possible problem with $${test}, diffs above"; \
	${RM} -f $${test}.tmp

runtest1_1:
	-@test=test1_1; \
	${MPIEXEC} -np 1 ./test1 -bv_type vecs -verbose 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest1_2:
	-@test=test1_2; \
	${MPIEXEC} -np 1 ./test1 -bv_type contiguous -verbose 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest1_3:
	-@test=test1_3; \
	${MPIEXEC} -np 1 ./test1 -bv_type svec -verbose 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest1_4:
	-@test=test1_4; \
	${MPIEXEC} -np 1 ./test1 -bv_type mat -verbose 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_1:
	-@test=test2_1; \
	${MPIEXEC} -np 1 ./test2 -bv_type vecs -bv_orthog_type cgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_2:
	-@test=test2_1; \
	${MPIEXEC} -np 1 ./test2 -bv_type contiguous -bv_orthog_type cgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_3:
	-@test=test2_1; \
	${MPIEXEC} -np 1 ./test2 -bv_type svec -bv_orthog_type cgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_4:
	-@test=test2_1; \
	${MPIEXEC} -np 1 ./test2 -bv_type mat -bv_orthog_type cgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_5:
	-@test=test2_1; \
	${MPIEXEC} -np 1 ./test2 -bv_type vecs -bv_orthog_type mgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_6:
	-@test=test2_1; \
	${MPIEXEC} -np 1 ./test2 -bv_type contiguous -bv_orthog_type mgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_7:
	-@test=test2_1; \
	${MPIEXEC} -np 1 ./test2 -bv_type svec -bv_orthog_type mgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_8:
	-@test=test2_1; \
	${MPIEXEC} -np 1 ./test2 -bv_type mat -bv_orthog_type mgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

