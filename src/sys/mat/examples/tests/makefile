#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

CFLAGS     =
FFLAGS     =
CPPFLAGS   =
FPPFLAGS   =
LOCDIR     = src/sys/mat/examples/tests/
EXAMPLESC  = test1.c
EXAMPLESF  =
MANSEC     = sys
TESTS      = test1

TESTEXAMPLES_C           = test1.PETSc runtest1_1 runtest1_2 test1.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_SYS_LIB}
	${RM} test1.o

#------------------------------------------------------------------------------------

runtest1_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test1 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 2 ./test1 > $${test}.tmp 2>&1; \
	${TESTCODE}

