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
LOCDIR     = src/sys/classes/fn/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c test7.c test8.c test9.c test10.c test11.c
EXAMPLESF  = test7f.F
MANSEC     = FN
TESTS      = test1 test2 test3 test4 test5 test6 test7 test7f test8 test9 test10 test11

TESTEXAMPLES_C       = test1.PETSc runtest1_1 test1.rm \
                       test2.PETSc runtest2_1 test2.rm \
                       test3.PETSc runtest3_1 runtest3_2 test3.rm \
                       test4.PETSc runtest4_1 test4.rm \
                       test5.PETSc runtest5_1 runtest5_2 test5.rm \
                       test6.PETSc runtest6_1 runtest6_2 test6.rm \
                       test7.PETSc runtest7_1 runtest7_2 test7.rm \
                       test8.PETSc runtest8_1 runtest8_2 test8.rm \
                       test9.PETSc runtest9_1 test9.rm \
                       test10.PETSc runtest10_1 test10.rm \
                       test11.PETSc runtest11_1 test11.rm
TESTEXAMPLES_FORTRAN = test7f.PETSc runtest7f_1 test7f.rm

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

test7f: test7f.o chkopts
	-${FLINKER} -o test7f test7f.o ${SLEPC_SYS_LIB}
	${RM} test7f.o

test8: test8.o chkopts
	-${CLINKER} -o test8 test8.o ${SLEPC_SYS_LIB}
	${RM} test8.o

test9: test9.o chkopts
	-${CLINKER} -o test9 test9.o ${SLEPC_SYS_LIB}
	${RM} test9.o

test10: test10.o chkopts
	-${CLINKER} -o test10 test10.o ${SLEPC_SYS_LIB}
	${RM} test10.o

test11: test11.o chkopts
	-${CLINKER} -o test11 test11.o ${SLEPC_SYS_LIB}
	${RM} test11.o

#------------------------------------------------------------------------------------

runtest1_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test1 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test2 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test3 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_2:
	-@${SETTEST}; check=test3_1; \
	${MPIEXEC} -n 1 ./test3 -inplace > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test4 -f1_fn_type exp -f1_fn_scale -2.5 -f2_fn_type rational -f2_fn_rational_numerator -1,1 -f2_fn_rational_denominator 1,-6,4 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test5 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_2:
	-@${SETTEST}; check=test5_1; \
	${MPIEXEC} -n 1 ./test5 -inplace > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test6 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_2:
	-@${SETTEST}; check=test6_1; \
	${MPIEXEC} -n 1 ./test6 -inplace > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test7 -tau .05 -eta 2 -n 100 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_2:
	-@${SETTEST}; check=test7_1; \
	${MPIEXEC} -n 1 ./test7 -tau .05 -eta 2 -n 100 -inplace > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test7f -tau .13 -eta 2 -n 19 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test8 -tau 0.9 -eta 0.5 -n 10 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_2:
	-@${SETTEST}; check=test8_1; \
	${MPIEXEC} -n 1 ./test8 -tau 0.9 -eta 0.5 -n 10 -inplace > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test9 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest10_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test10 -fn_phi_index 3 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test11 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_2:
	-@${SETTEST}; check=test11_1; \
	${MPIEXEC} -n 1 ./test11 -inplace > $${test}.tmp 2>&1; \
	${TESTCODE}

