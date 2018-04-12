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
LOCDIR     = src/sys/classes/st/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c
EXAMPLESF  =
MANSEC     = ST
TESTS      = test1 test2 test3 test4 test5

TESTEXAMPLES_C_NOTSINGLE = test1.PETSc runtest1_1 test1.rm \
                           test2.PETSc runtest2_1 test2.rm \
                           test3.PETSc runtest3_1 test3.rm \
                           test4.PETSc runtest4_1 runtest4_2 test4.rm \
                           test5.PETSc runtest5_1 test5.rm

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

test5: test5.o chkopts
	-${CLINKER} -o test5 test5.o ${SLEPC_SYS_LIB}
	${RM} test5.o

#------------------------------------------------------------------------------------

runtest1_1: runtest1_1_inplace runtest1_1_shell
runtest1_1_%:
	-@${SETTEST}; check=test1_1; matmode=$*; \
	${MPIEXEC} -n 1 ./test1 -st_matmode $$matmode > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1: runtest2_1_copy runtest2_1_inplace runtest2_1_shell
runtest2_1_%:
	-@${SETTEST}; check=test2_1; matmode=$*; \
	${MPIEXEC} -n 1 ./test2 -st_matmode $$matmode > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_1: runtest3_1_copy runtest3_1_inplace runtest3_1_shell
runtest3_1_%:
	-@${SETTEST}; check=test3_1; matmode=$*; \
	${MPIEXEC} -n 1 ./test3 -st_matmode $$matmode > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_1: runtest4_1_copy runtest4_1_shell
runtest4_1_%:
	-@${SETTEST}; check=test4_1; matmode=$*; \
	${MPIEXEC} -n 1 ./test4 -st_transform -st_matmode $$matmode > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_2: runtest4_2_copy runtest4_2_shell
runtest4_2_%:
	-@${SETTEST}; check=test4_2; matmode=$*; \
	${MPIEXEC} -n 1 ./test4 -st_matmode $$matmode > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_1: runtest5_1_copy runtest5_1_inplace runtest5_1_shell
runtest5_1_%:
	-@${SETTEST}; check=test5_1; matmode=$*; \
	if [ "$$matmode" = shell ]; then matmode="shell -ksp_type bcgs -pc_type jacobi"; fi; \
	${MPIEXEC} -n 1 ./test5 -st_matmode $$matmode > $${test}.tmp 2>&1; \
	${TESTCODE}

