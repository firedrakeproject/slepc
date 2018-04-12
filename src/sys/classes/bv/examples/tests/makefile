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
LOCDIR     = src/sys/classes/bv/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c test7.c test8.c test9.c test10.c \
             test11.c test12.c test13.c test14.c test15.c test16.c
EXAMPLESF  = test1f.F90
MANSEC     = BV
TESTS      = test1 test1f test2 test3 test4 test5 test6 test7 test8 test9 test10 test11 test12 \
             test13 test14 test15 test16

TESTEXAMPLES_C           = test1.PETSc runtest1_1 test1.rm \
                           test2.PETSc runtest2_1 runtest2_2 test2.rm \
                           test3.PETSc runtest3_1 runtest3_2 runtest3_3 test3.rm \
                           test4.PETSc runtest4_1 runtest4_2 test4.rm \
                           test5.PETSc runtest5_1 runtest5_2 test5.rm \
                           test6.PETSc runtest6_1 runtest6_2 runtest6_3 test6.rm \
                           test7.PETSc runtest7_1 runtest7_2 test7.rm \
                           test8.PETSc runtest8_1 runtest8_2 runtest8_3 runtest8_4 test8.rm \
                           test9.PETSc runtest9_1 runtest9_2 test9.rm \
                           test10.PETSc runtest10_1 test10.rm \
                           test11.PETSc runtest11_1 runtest11_2 runtest11_3 runtest11_4 \
                                        runtest11_5 runtest11_6 runtest11_7 runtest11_8 \
                                        runtest11_9 runtest11_10 runtest11_11 runtest11_13 \
                                        runtest11_14 runtest11_15 runtest11_16 runtest11_17 \
                                        runtest11_18 test11.rm \
                           test12.PETSc runtest12_1 test12.rm \
                           test13.PETSc runtest13_1 test13.rm \
                           test14.PETSc runtest14_1 test14.rm \
                           test15.PETSc runtest15_1 runtest15_2 test15.rm \
                           test16.PETSc runtest16_1 test16.rm
TESTEXAMPLES_C_NOTSINGLE = test2.PETSc runtest2_3 test2.rm \
                           test11.PETSc runtest11_12 test11.rm
TESTEXAMPLES_FORTRAN     = test1f.PETSc runtest1f_1 runtest1f_2 test1f.rm
TESTEXAMPLES_VECCUDA     = test1.PETSc runtest1_1_cuda test1.rm \
                           test2.PETSc runtest2_1_cuda runtest2_2_cuda test2.rm \
                           test3.PETSc runtest3_1_cuda test3.rm \
                           test4.PETSc runtest4_1_cuda test4.rm \
                           test5.PETSc runtest5_1_cuda runtest5_2_cuda test5.rm \
                           test6.PETSc runtest6_1_cuda runtest6_3_cuda test6.rm \
                           test7.PETSc runtest7_1_cuda test7.rm \
                           test8.PETSc runtest8_1_cuda runtest8_4_cuda test8.rm \
                           test9.PETSc runtest9_1_cuda test9.rm \
                           test10.PETSc runtest10_1_cuda test10.rm \
                           test11.PETSc runtest11_1_cuda runtest11_2_cuda runtest11_3_cuda \
                                        runtest11_4_cuda runtest11_5_cuda runtest11_6_cuda \
                                        runtest11_7_cuda runtest11_8_cuda runtest11_9_cuda \
                                        runtest11_10_cuda test11.rm \
                           test13.PETSc runtest13_1_cuda test13.rm \
                           test15.PETSc runtest15_1_cuda runtest15_2_cuda test15.rm

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

test13: test13.o chkopts
	-${CLINKER} -o test13 test13.o ${SLEPC_SYS_LIB}
	${RM} test13.o

test14: test14.o chkopts
	-${CLINKER} -o test14 test14.o ${SLEPC_SYS_LIB}
	${RM} test14.o

test15: test15.o chkopts
	-${CLINKER} -o test15 test15.o ${SLEPC_SYS_LIB}
	${RM} test15.o

test16: test16.o chkopts
	-${CLINKER} -o test16 test16.o ${SLEPC_SYS_LIB}
	${RM} test16.o

#------------------------------------------------------------------------------------

runtest1_1: runtest1_1_vecs runtest1_1_contiguous runtest1_1_svec runtest1_1_mat
runtest1_1_%:
	-@${SETTEST}; bv=$*; \
	${MPIEXEC} -n 1 ./test1 -bv_type $$bv -verbose > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1_1_cuda:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test1 -bv_type svec -vec_type cuda -verbose > $${test}.tmp 2>&1; \
	${TESTCODE_CUDA}

runtest1f_1: runtest1f_1_vecs runtest1f_1_contiguous runtest1f_1_svec runtest1f_1_mat
runtest1f_1_%:
	-@${SETTEST}; check=test1f_1; bv=$*; \
	${MPIEXEC} -n 1 ./test1f -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1f_2: runtest1f_2_vecs runtest1f_2_contiguous runtest1f_2_svec runtest1f_2_mat
runtest1f_2_%:
	-@${SETTEST}; check=test1f_1; bv=$*; \
	${MPIEXEC} -n 2 ./test1f -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1: runtest2_1_vecs runtest2_1_contiguous runtest2_1_svec runtest2_1_mat
runtest2_1_%:
	-@${SETTEST}; check=test2_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 1 ./test2 -bv_type $$bv -bv_orthog_type cgs > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_2: runtest2_2_vecs runtest2_2_contiguous runtest2_2_svec runtest2_2_mat
runtest2_2_%:
	-@${SETTEST}; check=test2_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 1 ./test2 -bv_type $$bv -bv_orthog_type mgs > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_3: runtest2_3_vecs runtest2_3_contiguous runtest2_3_svec runtest2_3_mat
runtest2_3_%:
	-@${SETTEST}; check=test2_3; bv=$*; \
	${MPIEXEC} -n 1 ./test2 -bv_type $$bv -condn 1e8 | ${GREP} -v "against" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_1: runtest3_1_vecs runtest3_1_contiguous runtest3_1_svec runtest3_1_svec_vecs runtest3_1_mat
runtest3_1_%:
	-@${SETTEST}; check=test3_1; bv=$*; \
	if [ "$$bv" = svec_vecs ]; then bv="svec -bv_matmult vecs"; \
	elif [ "$$bv" = cuda ]; then bv="svec -mat_type aijcusparse"; fi; \
	${MPIEXEC} -n 1 ./test3 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_2: runtest3_2_vecs runtest3_2_contiguous runtest3_2_svec runtest3_2_mat
runtest3_2_%:
	-@${SETTEST}; check=test3_1; bv=$*; \
	${MPIEXEC} -n 2 ./test3 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_3: runtest3_3_vecs runtest3_3_contiguous runtest3_3_svec runtest3_3_mat
runtest3_3_%:
	-@${SETTEST}; check=test3_1; bv=$*; \
	${MPIEXEC} -n 2 ./test3 -bv_type $$bv -bv_orthog_type mgs > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_1: runtest4_1_vecs runtest4_1_vecs_vmip runtest4_1_contiguous runtest4_1_svec runtest4_1_mat
runtest4_1_%:
	-@${SETTEST}; check=test4_1; bv=$*; \
	if [ "$$bv" = vecs_vmip ]; then bv="vecs -bv_vecs_vmip 1"; \
	elif [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 1 ./test4 -n 18 -kx 12 -ky 8 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_2: runtest4_2_vecs runtest4_2_contiguous runtest4_2_svec runtest4_2_mat
runtest4_2_%:
	-@${SETTEST}; check=test4_1; bv=$*; \
	${MPIEXEC} -n 1 ./test4 -n 18 -kx 12 -ky 8 -bv_type $$bv -trans > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_1: runtest5_1_vecs runtest5_1_contiguous runtest5_1_svec runtest5_1_mat
runtest5_1_%:
	-@${SETTEST}; check=test5_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -mat_type aijcusparse"; fi; \
	${MPIEXEC} -n 1 ./test5 -bv_orthog_refine always -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_2: runtest5_2_vecs runtest5_2_contiguous runtest5_2_svec runtest5_2_mat
runtest5_2_%:
	-@${SETTEST}; check=test5_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -mat_type aijcusparse"; fi; \
	${MPIEXEC} -n 1 ./test5 -bv_orthog_refine always -bv_type $$bv -bv_orthog_type mgs > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_1: runtest6_1_vecs runtest6_1_contiguous runtest6_1_svec runtest6_1_mat
runtest6_1_%:
	-@${SETTEST}; check=test6_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 1 ./test6 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_2: runtest6_2_vecs runtest6_2_contiguous runtest6_2_svec runtest6_2_mat
runtest6_2_%:
	-@${SETTEST}; check=test6_1; bv=$*; \
	${MPIEXEC} -n 2 ./test6 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_3: runtest6_3_vecs runtest6_3_contiguous runtest6_3_svec runtest6_3_mat
runtest6_3_%:
	-@${SETTEST}; check=test6_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 1 ./test6 -bv_type $$bv -bv_orthog_type mgs > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_1: runtest7_1_vecs runtest7_1_contiguous runtest7_1_svec runtest7_1_mat
runtest7_1_%:
	-@${SETTEST}; check=test7_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -mat_type aijcusparse"; fi; \
	${MPIEXEC} -n 1 ./test7 -bv_type $$bv -bv_matmult vecs | ${GREP} -v "Using method" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_2: runtest7_2_vecs runtest7_2_contiguous runtest7_2_svec runtest7_2_mat
runtest7_2_%:
	-@${SETTEST}; check=test7_1; bv=$*; \
	${MPIEXEC} -n 1 ./test7 -bv_type $$bv -bv_matmult mat | ${GREP} -v "Using method" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_1: runtest8_1_vecs runtest8_1_contiguous runtest8_1_svec runtest8_1_mat
runtest8_1_%:
	-@${SETTEST}; check=test8_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 1 ./test8 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_2: runtest8_2_vecs runtest8_2_contiguous runtest8_2_svec runtest8_2_mat
runtest8_2_%:
	-@${SETTEST}; check=test8_1; bv=$*; \
	${MPIEXEC} -n 1 ./test8 -bv_type $$bv -bv_orthog_refine never > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_3: runtest8_3_vecs runtest8_3_contiguous runtest8_3_svec runtest8_3_mat
runtest8_3_%:
	-@${SETTEST}; check=test8_1; bv=$*; \
	${MPIEXEC} -n 1 ./test8 -bv_type $$bv -bv_orthog_refine always > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_4: runtest8_4_vecs runtest8_4_contiguous runtest8_4_svec runtest8_4_mat
runtest8_4_%:
	-@${SETTEST}; check=test8_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 1 ./test8 -bv_type $$bv -bv_orthog_type mgs > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_1: runtest9_1_vecs runtest9_1_contiguous runtest9_1_svec runtest9_1_svec_vecs runtest9_1_mat
runtest9_1_%:
	-@${SETTEST}; check=test9_1; bv=$*; \
	if [ "$$bv" = svec_vecs ]; then bv="svec -bv_matmult vecs"; \
	elif [ "$$bv" = cuda ]; then bv="svec -mat_type aijcusparse"; fi; \
	${MPIEXEC} -n 1 ./test9 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_2: runtest9_2_vecs runtest9_2_contiguous runtest9_2_svec runtest9_2_svec_vecs runtest9_2_mat
runtest9_2_%:
	-@${SETTEST}; check=test9_1; bv=$*; \
	if [ "$$bv" = svec_vecs ]; then bv="svec -bv_matmult vecs"; fi; \
	${MPIEXEC} -n 2 ./test9 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest10_1: runtest10_1_vecs runtest10_1_contiguous runtest10_1_svec runtest10_1_mat
runtest10_1_%:
	-@${SETTEST}; check=test10_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test10 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_1: runtest11_1_vecs runtest11_1_contiguous runtest11_1_svec runtest11_1_mat
runtest11_1_%:
	-@${SETTEST}; check=test11_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block gs -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_2: runtest11_2_vecs runtest11_2_contiguous runtest11_2_svec runtest11_2_mat
runtest11_2_%:
	-@${SETTEST}; check=test11_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block chol -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_3: runtest11_3_vecs runtest11_3_contiguous runtest11_3_svec runtest11_3_mat
runtest11_3_%:
	-@${SETTEST}; check=test11_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block tsqr -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_4: runtest11_4_vecs runtest11_4_contiguous runtest11_4_svec runtest11_4_mat
runtest11_4_%:
	-@${SETTEST}; check=test11_4; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda -mat_type aijcusparse"; fi; \
	${MPIEXEC} -n 2 ./test11 -withb -bv_orthog_block gs -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_5: runtest11_5_vecs runtest11_5_contiguous runtest11_5_svec runtest11_5_mat
runtest11_5_%:
	-@${SETTEST}; check=test11_4; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda -mat_type aijcusparse"; fi; \
	${MPIEXEC} -n 2 ./test11 -withb -bv_orthog_block chol -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_6: runtest11_6_vecs runtest11_6_contiguous runtest11_6_svec runtest11_6_mat
runtest11_6_%:
	-@${SETTEST}; check=test11_6; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test11 -resid -bv_orthog_block gs -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_7: runtest11_7_vecs runtest11_7_contiguous runtest11_7_svec runtest11_7_mat
runtest11_7_%:
	-@${SETTEST}; check=test11_6; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test11 -resid -bv_orthog_block chol -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_8: runtest11_8_vecs runtest11_8_contiguous runtest11_8_svec runtest11_8_mat
runtest11_8_%:
	-@${SETTEST}; check=test11_6; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test11 -resid -bv_orthog_block tsqr -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_9: runtest11_9_vecs runtest11_9_contiguous runtest11_9_svec runtest11_9_mat
runtest11_9_%:
	-@${SETTEST}; check=test11_9; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda -mat_type aijcusparse"; fi; \
	${MPIEXEC} -n 2 ./test11 -resid -withb -bv_orthog_block gs -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_10: runtest11_10_vecs runtest11_10_contiguous runtest11_10_svec runtest11_10_mat
runtest11_10_%:
	-@${SETTEST}; check=test11_9; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda -mat_type aijcusparse"; fi; \
	${MPIEXEC} -n 2 ./test11 -resid -withb -bv_orthog_block chol -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_11: runtest11_11_vecs runtest11_11_contiguous runtest11_11_svec runtest11_11_mat
runtest11_11_%:
	-@${SETTEST}; check=test11_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 7 ./test11 -bv_orthog_block tsqr -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_12: runtest11_12_vecs runtest11_12_contiguous runtest11_12_svec runtest11_12_mat
runtest11_12_%:
	-@${SETTEST}; check=test11_12; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 9 ./test11 -resid -n 180 -l 0 -k 7 -bv_orthog_block tsqr -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_13: runtest11_13_vecs runtest11_13_contiguous runtest11_13_svec runtest11_13_mat
runtest11_13_%:
	-@${SETTEST}; check=test11_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block tsqrchol -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_14: runtest11_14_vecs runtest11_14_contiguous runtest11_14_svec runtest11_14_mat
runtest11_14_%:
	-@${SETTEST}; check=test11_6; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test11 -resid -bv_orthog_block tsqrchol -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_15: runtest11_15_vecs runtest11_15_contiguous runtest11_15_svec runtest11_15_mat
runtest11_15_%:
	-@${SETTEST}; check=test11_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test11 -bv_orthog_block svqb -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_16: runtest11_16_vecs runtest11_16_contiguous runtest11_16_svec runtest11_16_mat
runtest11_16_%:
	-@${SETTEST}; check=test11_4; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test11 -withb -bv_orthog_block svqb -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_17: runtest11_17_vecs runtest11_17_contiguous runtest11_17_svec runtest11_17_mat
runtest11_17_%:
	-@${SETTEST}; check=test11_6; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test11 -resid -bv_orthog_block svqb -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_18: runtest11_18_vecs runtest11_18_contiguous runtest11_18_svec runtest11_18_mat
runtest11_18_%:
	-@${SETTEST}; check=test11_9; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 2 ./test11 -resid -withb -bv_orthog_block svqb -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest12_1: runtest12_1_vecs runtest12_1_contiguous runtest12_1_svec runtest12_1_mat
runtest12_1_%:
	-@${SETTEST}; check=test12_1; bv=$*; \
	${MPIEXEC} -n 1 ./test12 -bv_orthog_block gs -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

#runtest12_2:
#	-@test=test12_1; \
#	${MPIEXEC} -n 1 ./test12 -bv_orthog_block chol 2>&1 > $${test}.tmp; \
#	${TESTCODE}

runtest13_1: runtest13_1_vecs runtest13_1_contiguous runtest13_1_svec runtest13_1_mat
runtest13_1_%:
	-@${SETTEST}; check=test13_1; bv=$*; \
	if [ "$$bv" = cuda ]; then bv="svec -vec_type cuda"; fi; \
	${MPIEXEC} -n 1 ./test13 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest14_1: runtest14_1_vecs runtest14_1_contiguous runtest14_1_svec runtest14_1_mat
runtest14_1_%:
	-@${SETTEST}; check=test14_1; bv=$*; \
	${MPIEXEC} -n 2 ./test14 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest15_1: runtest15_1_vecs runtest15_1_contiguous runtest15_1_svec runtest15_1_mat
runtest15_1_%:
	-@${SETTEST}; check=test15_1; bv=$*; \
	${MPIEXEC} -n 2 ./test15 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest15_1_cuda:
	-@${SETTEST}; check=test15_1; \
	${MPIEXEC} -n 2 ./test15 -bv_type svec -vec_type cuda > $${test}.tmp 2>&1; \
	${TESTCODE_CUDA}

runtest15_2: runtest15_2_vecs runtest15_2_contiguous runtest15_2_svec runtest15_2_mat
runtest15_2_%:
	-@${SETTEST}; check=test15_2; bv=$*; \
	${MPIEXEC} -n 2 ./test15 -nc 2 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest15_2_cuda:
	-@${SETTEST}; check=test15_2; \
	${MPIEXEC} -n 2 ./test15 -nc 2 -bv_type svec -vec_type cuda > $${test}.tmp 2>&1; \
	${TESTCODE_CUDA}

runtest16_1: runtest16_1_vecs runtest16_1_contiguous runtest16_1_svec runtest16_1_mat
runtest16_1_%:
	-@${SETTEST}; check=test16_1; bv=$*; \
	${MPIEXEC} -n 2 ./test16 -bv_type $$bv > $${test}.tmp 2>&1; \
	${TESTCODE}

