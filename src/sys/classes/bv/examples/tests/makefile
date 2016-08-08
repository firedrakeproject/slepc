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

TESTEXAMPLES_C       = test1.PETSc runtest1_1 test1.rm \
                       test2.PETSc runtest2_1 runtest2_2 test2.rm \
                       test3.PETSc runtest3_1 test3.rm \
                       test4.PETSc runtest4_1 test4.rm \
                       test5.PETSc runtest5_1 test5.rm \
                       test6.PETSc runtest6_1 test6.rm \
                       test7.PETSc runtest7_1 test7.rm \
                       test8.PETSc runtest8_1 test8.rm \
                       test9.PETSc runtest9_1 test9.rm \
                       test10.PETSc runtest10_1 test10.rm \
                       test11.PETSc runtest11_1 runtest11_2 test11.rm \
                       test12.PETSc runtest12_1 test12.rm

TESTEXAMPLES_VECCUDA = test1.PETSc runtest1_1_cuda test1.rm \
                       test2.PETSc runtest2_1_cuda runtest2_2_cuda test2.rm \
                       test3.PETSc runtest3_1_cuda test3.rm \
                       test4.PETSc runtest4_1_cuda test4.rm \
                       test6.PETSc runtest6_1_cuda test6.rm \
                       test7.PETSc runtest7_1_cuda test7.rm

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

runtest1_1: runtest1_1_vecs runtest1_1_contiguous runtest1_1_svec runtest1_1_mat
runtest1_1_%:
	-@${SETTEST}; bv=$*; \
	${MPIEXEC} -n 1 ./test1 -bv_type $$bv -verbose > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1_1_cuda:
	-@${SETTEST}; check=test1_1_svec; \
	${MPIEXEC} -n 1 ./test1 -bv_type svec -vec_type cuda -verbose > $${test}.tmp 2>&1; \
	${TESTCODE_CUDA}

runtest2_1: runtest2_1_vecs runtest2_1_contiguous runtest2_1_svec runtest2_1_mat
runtest2_1_%:
	-@${SETTEST}; check=test2_1; bv=$*; \
	${MPIEXEC} -n 1 ./test2 -bv_type $$bv -bv_orthog_type cgs > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1_cuda:
	-@${SETTEST}; check=test2_1; \
	${MPIEXEC} -n 1 ./test2 -bv_type svec -bv_orthog_type cgs -vec_type cuda > $${test}.tmp 2>&1; \
	${TESTCODE_CUDA}

runtest2_2: runtest2_2_vecs runtest2_2_contiguous runtest2_2_svec runtest2_2_mat
runtest2_2_%:
	-@${SETTEST}; check=test2_1; bv=$*; \
	${MPIEXEC} -n 1 ./test2 -bv_type $$bv -bv_orthog_type mgs > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_2_cuda:
	-@${SETTEST}; check=test2_1; \
	${MPIEXEC} -n 1 ./test2 -bv_type svec -bv_orthog_type mgs -vec_type cuda > $${test}.tmp 2>&1; \
	${TESTCODE_CUDA}

runtest3_1: runtest3_1_vecs runtest3_1_contiguous runtest3_1_svec runtest3_1_mat
runtest3_1_%:
	-@${SETTEST}; check=test3_1; bv=$*; \
	${MPIEXEC} -n 1 ./test3 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_1_cuda:
	-@${SETTEST}; check=test3_1; \
	${MPIEXEC} -n 1 ./test3 -bv_type svec -mat_type aijcusparse > $${test}.tmp 2>&1; \
	${TESTCODE_CUDA}

runtest4_1: runtest4_1_vecs runtest4_1_contiguous runtest4_1_svec runtest4_1_svec_trans runtest4_1_mat
runtest4_1_%:
	-@${SETTEST}; check=test4_1; bv=$*; \
	if [ "$$bv" = svec_trans ]; then bv="svec -trans"; fi; \
	${MPIEXEC} -n 1 ./test4 -n 18 -kx 12 -ky 8 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_1_cuda:
	-@${SETTEST}; check=test4_1; \
	${MPIEXEC} -n 1 ./test4 -n 18 -kx 12 -ky 8 -bv_type svec -trans -vec_type cuda > $${test}.tmp 2>&1; \
	${TESTCODE_CUDA}

runtest5_1: runtest5_1_vecs runtest5_1_contiguous runtest5_1_svec runtest5_1_mat
runtest5_1_%:
	-@${SETTEST}; check=test5_1; bv=$*; \
	${MPIEXEC} -n 1 ./test5 -bv_orthog_refine always -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_1: runtest6_1_vecs runtest6_1_contiguous runtest6_1_svec runtest6_1_mat
runtest6_1_%:
	-@${SETTEST}; check=test6_1; bv=$*; \
	${MPIEXEC} -n 1 ./test6 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_1_cuda:
	-@${SETTEST}; check=test6_1; \
	${MPIEXEC} -n 1 ./test6 -bv_type svec -vec_type cuda > $${test}.tmp 2>&1; \
	${TESTCODE_CUDA}

runtest7_1: runtest7_1_vecs runtest7_1_contiguous runtest7_1_svec runtest7_1_mat
runtest7_1_%:
	-@${SETTEST}; check=test7_1; bv=$*; \
	${MPIEXEC} -n 1 ./test7 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_1_cuda:
	-@${SETTEST}; check=test7_1; \
	${MPIEXEC} -n 1 ./test7 -bv_type svec -mat_type aijcusparse > $${test}.tmp 2>&1; \
	${TESTCODE_CUDA}

runtest8_1: runtest8_1_vecs runtest8_1_contiguous runtest8_1_svec runtest8_1_mat
runtest8_1_%:
	-@${SETTEST}; check=test8_1; bv=$*; \
	${MPIEXEC} -n 1 ./test8 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_1: runtest9_1_vecs runtest9_1_contiguous runtest9_1_svec runtest9_1_mat
runtest9_1_%:
	-@${SETTEST}; check=test9_1; bv=$*; \
	${MPIEXEC} -n 1 ./test9 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest10_1: runtest10_1_vecs runtest10_1_contiguous runtest10_1_svec runtest10_1_mat
runtest10_1_%:
	-@${SETTEST}; check=test10_1; bv=$*; \
	${MPIEXEC} -n 2 ./test10 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_1: runtest11_1_vecs runtest11_1_contiguous runtest11_1_svec runtest11_1_mat
runtest11_1_%:
	-@${SETTEST}; check=test11_1; bv=$*; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block gs -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_2: runtest11_2_vecs runtest11_2_contiguous runtest11_2_svec runtest11_2_mat
runtest11_2_%:
	-@${SETTEST}; check=test11_1; bv=$*; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block chol -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest12_1: runtest12_1_vecs runtest12_1_contiguous runtest12_1_svec runtest12_1_mat
runtest12_1_%:
	-@${SETTEST}; check=test12_1; bv=$*; \
	${MPIEXEC} -n 1 ./test12 -bv_orthog_block gs -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

#runtest12_2:
#	-@test=test12_1; \
#	${MPIEXEC} -n 1 ./test12 -bv_orthog_block chol mat 2>&1 > $${test}.tmp; \
#	${TESTCODE}

