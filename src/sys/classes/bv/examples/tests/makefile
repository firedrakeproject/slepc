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
LOCDIR     = src/sys/classes/bv/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c test7.c test8.c test9.c test10.c test11.c test12.c
EXAMPLESF  =
MANSEC     = BV
TESTS      = test1 test2 test3 test4 test5 test6 test7 test8 test9 test10 test11 test12

TESTEXAMPLES_C       = test1.PETSc runtest1_1 runtest1_2 runtest1_3 runtest1_4 test1.rm \
                       test2.PETSc runtest2_1 runtest2_2 runtest2_3 runtest2_4 \
                                   runtest2_5 runtest2_6 runtest2_7 runtest2_8 test2.rm \
                       test3.PETSc runtest3_1 runtest3_2 runtest3_3 runtest3_4 test3.rm \
                       test4.PETSc runtest4_1 runtest4_2 runtest4_3 runtest4_4 test4.rm \
                       test5.PETSc runtest5_1 runtest5_2 runtest5_3 runtest5_4 test5.rm \
                       test6.PETSc runtest6_1 runtest6_2 runtest6_3 runtest6_4 test6.rm \
                       test7.PETSc runtest7_1 runtest7_2 runtest7_3 runtest7_4 test7.rm \
                       test8.PETSc runtest8_1 runtest8_2 runtest8_3 runtest8_4 test8.rm \
                       test9.PETSc runtest9_1 runtest9_2 runtest9_3 runtest9_4 test9.rm \
                       test10.PETSc runtest10_1 runtest10_2 runtest10_3 runtest10_4 test10.rm \
                       test11.PETSc runtest11_1 runtest11_2 runtest11_3 runtest11_4 \
                                    runtest11_5 runtest11_6 runtest11_7 runtest11_8 test11.rm \
                       test12.PETSc runtest12_1 test12.rm

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

test10: test10.o chkopts
	-${CLINKER} -o test10 test10.o ${SLEPC_SYS_LIB}
	${RM} test10.o

test11: test11.o chkopts
	-${CLINKER} -o test11 test11.o ${SLEPC_SYS_LIB}
	${RM} test11.o

test12: test12.o chkopts
	-${CLINKER} -o test12 test12.o ${SLEPC_SYS_LIB}
	${RM} test12.o

#------------------------------------------------------------------------------------

TESTCODE = \
	[ x${SAVE_OUTPUT} = xyes ] && cp $${test}.tmp output/$${test}.out; \
	${DIFF} output/$${test}.out $${test}.tmp || \
	echo "Possible problem with $${test}, diffs above"; \
	${RM} -f $${test}.tmp

runtest1_1:
	-@test=test1_1; \
	${MPIEXEC} -n 1 ./test1 -bv_type vecs -verbose 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest1_2:
	-@test=test1_2; \
	${MPIEXEC} -n 1 ./test1 -bv_type contiguous -verbose 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest1_3:
	-@test=test1_3; \
	${MPIEXEC} -n 1 ./test1 -bv_type svec -verbose 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest1_4:
	-@test=test1_4; \
	${MPIEXEC} -n 1 ./test1 -bv_type mat -verbose 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_1:
	-@test=test2_1; \
	${MPIEXEC} -n 1 ./test2 -bv_type vecs -bv_orthog_type cgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_2:
	-@test=test2_1; \
	${MPIEXEC} -n 1 ./test2 -bv_type contiguous -bv_orthog_type cgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_3:
	-@test=test2_1; \
	${MPIEXEC} -n 1 ./test2 -bv_type svec -bv_orthog_type cgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_4:
	-@test=test2_1; \
	${MPIEXEC} -n 1 ./test2 -bv_type mat -bv_orthog_type cgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_5:
	-@test=test2_1; \
	${MPIEXEC} -n 1 ./test2 -bv_type vecs -bv_orthog_type mgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_6:
	-@test=test2_1; \
	${MPIEXEC} -n 1 ./test2 -bv_type contiguous -bv_orthog_type mgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_7:
	-@test=test2_1; \
	${MPIEXEC} -n 1 ./test2 -bv_type svec -bv_orthog_type mgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest2_8:
	-@test=test2_1; \
	${MPIEXEC} -n 1 ./test2 -bv_type mat -bv_orthog_type mgs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest3_1:
	-@test=test3_1; \
	${MPIEXEC} -n 1 ./test3 -bv_type vecs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest3_2:
	-@test=test3_1; \
	${MPIEXEC} -n 1 ./test3 -bv_type contiguous 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest3_3:
	-@test=test3_1; \
	${MPIEXEC} -n 1 ./test3 -bv_type svec 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest3_4:
	-@test=test3_1; \
	${MPIEXEC} -n 1 ./test3 -bv_type mat 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest4_1:
	-@test=test4_1; \
	${MPIEXEC} -n 1 ./test4 -n 18 -kx 12 -ky 8 -bv_type vecs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest4_2:
	-@test=test4_1; \
	${MPIEXEC} -n 1 ./test4 -n 18 -kx 12 -ky 8 -bv_type svec 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest4_3:
	-@test=test4_1; \
	${MPIEXEC} -n 1 ./test4 -n 18 -kx 12 -ky 8 -bv_type svec -trans 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest4_4:
	-@test=test4_1; \
	${MPIEXEC} -n 1 ./test4 -n 18 -kx 12 -ky 8 -bv_type mat -trans 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest5_1:
	-@test=test5_1; \
	${MPIEXEC} -n 1 ./test5 -bv_orthog_refine always -bv_type vecs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest5_2:
	-@test=test5_1; \
	${MPIEXEC} -n 1 ./test5 -bv_orthog_refine always -bv_type contiguous 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest5_3:
	-@test=test5_1; \
	${MPIEXEC} -n 1 ./test5 -bv_orthog_refine always -bv_type svec 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest5_4:
	-@test=test5_1; \
	${MPIEXEC} -n 1 ./test5 -bv_orthog_refine always -bv_type mat 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest6_1:
	-@test=test6_1; \
	${MPIEXEC} -n 1 ./test6 -bv_type vecs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest6_2:
	-@test=test6_1; \
	${MPIEXEC} -n 1 ./test6 -bv_type contiguous 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest6_3:
	-@test=test6_1; \
	${MPIEXEC} -n 1 ./test6 -bv_type svec 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest6_4:
	-@test=test6_1; \
	${MPIEXEC} -n 1 ./test6 -bv_type mat 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest7_1:
	-@test=test7_1; \
	${MPIEXEC} -n 1 ./test7 -bv_type vecs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest7_2:
	-@test=test7_1; \
	${MPIEXEC} -n 1 ./test7 -bv_type contiguous 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest7_3:
	-@test=test7_1; \
	${MPIEXEC} -n 1 ./test7 -bv_type svec 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest7_4:
	-@test=test7_1; \
	${MPIEXEC} -n 1 ./test7 -bv_type mat 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest8_1:
	-@test=test8_1; \
	${MPIEXEC} -n 1 ./test8 -bv_type vecs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest8_2:
	-@test=test8_1; \
	${MPIEXEC} -n 1 ./test8 -bv_type contiguous 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest8_3:
	-@test=test8_1; \
	${MPIEXEC} -n 1 ./test8 -bv_type svec 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest8_4:
	-@test=test8_1; \
	${MPIEXEC} -n 1 ./test8 -bv_type mat 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest9_1:
	-@test=test9_1; \
	${MPIEXEC} -n 1 ./test9 -bv_type vecs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest9_2:
	-@test=test9_1; \
	${MPIEXEC} -n 1 ./test9 -bv_type contiguous 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest9_3:
	-@test=test9_1; \
	${MPIEXEC} -n 1 ./test9 -bv_type svec 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest9_4:
	-@test=test9_1; \
	${MPIEXEC} -n 1 ./test9 -bv_type mat 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest10_1:
	-@test=test10_1; \
	${MPIEXEC} -n 2 ./test10 -bv_type vecs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest10_2:
	-@test=test10_1; \
	${MPIEXEC} -n 2 ./test10 -bv_type contiguous 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest10_3:
	-@test=test10_1; \
	${MPIEXEC} -n 2 ./test10 -bv_type svec 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest10_4:
	-@test=test10_1; \
	${MPIEXEC} -n 2 ./test10 -bv_type mat 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest11_1:
	-@test=test11_1; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block gs -bv_type vecs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest11_2:
	-@test=test11_1; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block gs -bv_type contiguous 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest11_3:
	-@test=test11_1; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block gs -bv_type svec 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest11_4:
	-@test=test11_1; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block gs -bv_type mat 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest11_5:
	-@test=test11_1; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block chol -bv_type vecs 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest11_6:
	-@test=test11_1; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block chol -bv_type contiguous 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest11_7:
	-@test=test11_1; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block chol -bv_type svec 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest11_8:
	-@test=test11_1; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block chol -bv_type mat 2>&1 > $${test}.tmp; \
	${TESTCODE}

runtest12_1:
	-@test=test12_1; \
	${MPIEXEC} -n 1 ./test12 -bv_orthog_block gs mat 2>&1 > $${test}.tmp; \
	${TESTCODE}

#runtest12_2:
#	-@test=test12_1; \
#	${MPIEXEC} -n 1 ./test12 -bv_orthog_block chol mat 2>&1 > $${test}.tmp; \
#	${TESTCODE}

