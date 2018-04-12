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
LOCDIR     = src/sys/classes/ds/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c test7.c test8.c test9.c \
             test12.c test13.c test15.c test16.c test17.c test18.c
EXAMPLESF  = test14f.F90
MANSEC     = DS
TESTS      = test1 test2 test3 test4 test5 test6 test7 test8 test9 test12 test13 \
             test14f test15 test16 test17 test18

TESTEXAMPLES_C                     = test2.PETSc runtest2_1 test2.rm \
                                     test12.PETSc runtest12_1 test12.rm \
                                     test16.PETSc runtest16_1 test16.rm 
TESTEXAMPLES_C_NOTSINGLE           = test3.PETSc runtest3_1 test3.rm \
                                     test5.PETSc runtest5_1 test5.rm \
                                     test6.PETSc runtest6_1 test6.rm \
                                     test7.PETSc runtest7_1 test7.rm \
                                     test8.PETSc runtest8_1 test8.rm \
                                     test9.PETSc runtest9_1 test9.rm \
                                     test15.PETSc runtest15_1 test15.rm
TESTEXAMPLES_C_NOCOMPLEX           = test1.PETSc runtest1_1 test1.rm \
                                     test4.PETSc runtest4_1 test4.rm \
                                     test17.PETSc runtest17_1 test17.rm \
                                     test18.PETSc runtest18_1 runtest18_2 test18.rm
TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE = test13.PETSc runtest13_1 test13.rm
TESTEXAMPLES_FORTRAN_NOCOMPLEX     = test14f.PETSc runtest14f_1 test14f.rm

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

test6: test6.o chkopts
	-${CLINKER} -o test6 test6.o ${SLEPC_SYS_LIB}
	${RM} test6.o

test7: test7.o chkopts
	-${CLINKER} -o test7 test7.o ${SLEPC_SYS_LIB}
	${RM} test7.o

test8: test8.o chkopts
	-${CLINKER} -o test8 test8.o ${SLEPC_SYS_LIB}
	${RM} test8.o

test9: test9.o chkopts
	-${CLINKER} -o test9 test9.o ${SLEPC_SYS_LIB}
	${RM} test9.o

test12: test12.o chkopts
	-${CLINKER} -o test12 test12.o ${SLEPC_SYS_LIB}
	${RM} test12.o

test13: test13.o chkopts
	-${CLINKER} -o test13 test13.o ${SLEPC_SYS_LIB}
	${RM} test13.o

test14f: test14f.o chkopts
	-${FLINKER} -o test14f test14f.o ${SLEPC_SYS_LIB}
	${RM} test14f.o

test15: test15.o chkopts
	-${CLINKER} -o test15 test15.o ${SLEPC_SYS_LIB}
	${RM} test15.o

test16: test16.o chkopts
	-${CLINKER} -o test16 test16.o ${SLEPC_SYS_LIB}
	${RM} test16.o

test17: test17.o chkopts
	-${CLINKER} -o test17 test17.o ${SLEPC_SYS_LIB}
	${RM} test17.o

test18: test18.o chkopts
	-${CLINKER} -o test18 test18.o ${SLEPC_SYS_LIB}
	${RM} test18.o

#------------------------------------------------------------------------------------

runtest1_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test1 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1: runtest2_1_qr runtest2_1_mrrr runtest2_1_dc
runtest2_1_%:
	-@${SETTEST}; check=test2_1; method=$*; \
	if [ "$$method" = qr ]; then method="-ds_method 0"; \
	elif [ "$$method" = mrrr ]; then method="-ds_method 1"; \
	elif [ "$$method" = dc ]; then method="-ds_method 2"; fi; \
	${MPIEXEC} -n 1 ./test2 -n 12 $$method | ${GREP} -v "solving the problem" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_1: runtest3_1_qr runtest3_1_mrrr runtest3_1_dc
runtest3_1_%:
	-@${SETTEST}; check=test3_1; method=$*; \
	if [ "$$method" = qr ]; then method="-ds_method 0"; \
	elif [ "$$method" = mrrr ]; then method="-ds_method 1"; \
	elif [ "$$method" = dc ]; then method="-ds_method 2"; fi; \
	${MPIEXEC} -n 1 ./test3 -n 9 $$method | ${GREP} -v "solving the problem" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test4 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_1: runtest5_1_hr runtest5_1_qr_ii runtest5_1_qr
runtest5_1_%:
	-@${SETTEST}; check=test5_1; method=$*; \
	if [ "$$method" = hr ]; then method="-ds_method 0"; \
	elif [ "$$method" = qr_ii ]; then method="-ds_method 1"; \
	elif [ "$$method" = qr ]; then method="-ds_method 2"; fi; \
	${MPIEXEC} -n 1 ./test5 $$method | ${GREP} -v "solving the problem" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_1: runtest6_1_hr runtest6_1_qr_ii runtest6_1_qr
runtest6_1_%:
	-@${SETTEST}; check=test6_1; method=$*; \
	if [ "$$method" = hr ]; then method="-ds_method 0"; \
	elif [ "$$method" = qr_ii ]; then method="-ds_method 1"; \
	elif [ "$$method" = qr ]; then method="-ds_method 2"; fi; \
	${MPIEXEC} -n 1 ./test6 $$method | ${GREP} -v "solving the problem" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test7 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test8 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test9 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest12_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test12 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest13_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test13 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest14f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test14f > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest15_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test15 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest16_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test16 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest17_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test17 -n 7 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest18_1: runtest18_1_1 runtest18_1_2 runtest18_1_3
runtest18_1_%:
	-@${SETTEST}; check=test18_1; np=$*; \
	${MPIEXEC} -n $$np ./test18 -ds_parallel redundant > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest18_2: runtest18_2_1 runtest18_2_2 runtest18_2_3
runtest18_2_%:
	-@${SETTEST}; check=test18_1; np=$*; \
	${MPIEXEC} -n $$np ./test18 -ds_parallel synchronized > $${test}.tmp 2>&1; \
	${TESTCODE}

