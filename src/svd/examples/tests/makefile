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
LOCDIR     = src/svd/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c test7.c test8.c test9.c
EXAMPLESF  = test4f.F
MANSEC     = SVD
TESTS      = test1 test2 test3 test4 test4f test5 test6 test7 test8 test9

TESTEXAMPLES_C           = test1.PETSc runtest1_1 test1.rm \
                           test4.PETSc runtest4_1 runtest4_3 test4.rm \
                           test5.PETSc runtest5_1 test5.rm \
                           test6.PETSc runtest6_1 test6.rm \
                           test8.PETSc runtest8_1 test8.rm
TESTEXAMPLES_C_DOUBLE    = test2.PETSc runtest2_1 test2.rm
TESTEXAMPLES_C_NOTSINGLE = test3.PETSc runtest3_1 runtest3_2 runtest3_4 test3.rm \
                           test4.PETSc runtest4_1_lapack test4.rm \
                           test7.PETSc runtest7_1 test7.rm \
                           test9.PETSc runtest9_1 test9.rm
TESTEXAMPLES_C_NOCOMPLEX = test3.PETSc runtest3_2_cross_exp test3.rm \
                           test4.PETSc runtest4_1_cross_exp_imp test4.rm
TESTEXAMPLES_FORTRAN     = test4f.PETSc runtest4f_1 test4f.rm
TESTEXAMPLES_VECCUDA     = test3.PETSc runtest3_3_cuda test3.rm \
                           test4.PETSc runtest4_2_cuda test4.rm \
                           test6.PETSc runtest6_2_cuda test6.rm \
                           test7.PETSc runtest7_2_cuda test7.rm
TESTEXAMPLES_PRIMME      = test1.PETSc runtest1_1_primme test1.rm \
                           test2.PETSc runtest2_1_primme test2.rm \
                           test3.PETSc runtest3_1_primme test3.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_SVD_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_SVD_LIB}
	${RM} test2.o

test3: test3.o chkopts
	-${CLINKER} -o test3 test3.o ${SLEPC_SVD_LIB}
	${RM} test3.o

test4: test4.o chkopts
	-${CLINKER} -o test4 test4.o ${SLEPC_SVD_LIB}
	${RM} test4.o

test4f: test4f.o chkopts
	-${FLINKER} -o test4f test4f.o ${SLEPC_SVD_LIB}
	${RM} test4f.o

test5: test5.o chkopts
	-${CLINKER} -o test5 test5.o ${SLEPC_SVD_LIB}
	${RM} test5.o

test6: test6.o chkopts
	-${CLINKER} -o test6 test6.o ${SLEPC_SVD_LIB}
	${RM} test6.o

test7: test7.o chkopts
	-${CLINKER} -o test7 test7.o ${SLEPC_SVD_LIB}
	${RM} test7.o

test8: test8.o chkopts
	-${CLINKER} -o test8 test8.o ${SLEPC_SVD_LIB}
	${RM} test8.o

test9: test9.o chkopts
	-${CLINKER} -o test9 test9.o ${SLEPC_SVD_LIB}
	${RM} test9.o

#------------------------------------------------------------------------------------

runtest1_1: runtest1_1_lanczos runtest1_1_trlanczos runtest1_1_cross runtest1_1_cross_gd runtest1_1_cyclic runtest1_1_cyclic_gd runtest1_1_lapack
runtest1_1_%:
	-@${SETTEST}; check=test1_1; svd=$*; \
	if [ "$$svd" = cross_gd ]; then svd="cross -epstype gd"; \
	elif [ "$$svd" = cyclic_gd ]; then svd="cyclic -epstype gd"; fi; \
	${MPIEXEC} -n 1 ./test1 -type $$svd >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1: runtest2_1_lanczos runtest2_1_trlanczos runtest2_1_cross runtest2_1_cyclic runtest2_1_lapack
runtest2_1_%:
	-@${SETTEST}; check=test2_1; svd=$*; \
	if [ "${PETSC_SCALAR}" = "complex" ]; then check=$${check}_complex; fi; \
	${MPIEXEC} -n 1 ./test2 -svd_type $$svd -svd_nsv 7 > $${test}.tmp 2>&1;    \
	${TESTCODE}

runtest3_1: runtest3_1_lanczos runtest3_1_lanczos_one runtest3_1_trlanczos runtest3_1_trlanczos_one runtest3_1_trlanczos_one_mgs runtest3_1_trlanczos_one_alway runtest3_1_cross runtest3_1_cross_exp runtest3_1_cyclic runtest3_1_cyclic_exp runtest3_1_lapack
runtest3_1_%:
	-@${SETTEST}; check=test3_1; svd=$*; \
	if [ "$$svd" = lanczos_one ]; then svd="lanczos -svd_lanczos_oneside"; \
	elif [ "$$svd" = trlanczos_one ]; then svd="trlanczos -svd_trlanczos_oneside"; \
	elif [ "$$svd" = trlanczos_one_mgs ]; then svd="trlanczos -svd_trlanczos_oneside -bv_orthog_type mgs"; \
	elif [ "$$svd" = trlanczos_one_alway ]; then svd="trlanczos -svd_trlanczos_oneside -bv_orthog_refine always"; \
	elif [ "$$svd" = cross_exp ]; then svd="cross -svd_cross_explicitmatrix"; \
	elif [ "$$svd" = cyclic_exp ]; then svd="cyclic -svd_cyclic_explicitmatrix"; fi; \
	${MPIEXEC} -n 1 ./test3 -svd_type $$svd -svd_nsv 4 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_2: runtest3_2_lanczos runtest3_2_lanczos_one runtest3_2_trlanczos runtest3_2_trlanczos_one runtest3_2_trlanczos_one_mgs runtest3_2_trlanczos_one_alway runtest3_2_trlanczos_one_mgs runtest3_2_cross runtest3_2_cyclic runtest3_2_lapack
runtest3_2_%:
	-@${SETTEST}; check=test3_1; svd=$*; \
	if [ "$$svd" = lanczos_one ]; then svd="lanczos -svd_lanczos_oneside"; \
	elif [ "$$svd" = trlanczos_one ]; then svd="trlanczos -svd_trlanczos_oneside"; \
	elif [ "$$svd" = trlanczos_one_mgs ]; then svd="trlanczos -svd_trlanczos_oneside -bv_orthog_type mgs"; \
	elif [ "$$svd" = trlanczos_one_alway ]; then svd="trlanczos -svd_trlanczos_oneside -bv_orthog_refine always"; \
	elif [ "$$svd" = cross_exp ]; then svd="cross -svd_cross_explicitmatrix"; fi; \
	${MPIEXEC} -n 1 ./test3 -svd_implicittranspose -svd_type $$svd -svd_nsv 4 -svd_tol 1e-5 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_3_cuda: runtest3_3_cuda_lanczos runtest3_3_cuda_lanczos_one runtest3_3_cuda_trlanczos runtest3_3_cuda_trl_one runtest3_3_cuda_trl_one_mgs runtest3_3_cuda_trl_one_alway runtest3_3_cuda_cross runtest3_3_cuda_cyclic runtest3_3_cuda_cyclic_exp
runtest3_3_cuda_%:
	-@${SETTEST}; check=test3_1; svd=$*; \
	if [ "$$svd" = lanczos_one ]; then svd="lanczos -svd_lanczos_oneside"; \
	elif [ "$$svd" = cyclic_exp ]; then svd="cyclic -svd_cyclic_explicitmatrix"; \
	elif [ "$$svd" = trl_one ]; then svd="trlanczos -svd_trlanczos_oneside"; \
	elif [ "$$svd" = trl_one_mgs ]; then svd="trlanczos -svd_trlanczos_oneside -bv_orthog_type mgs"; \
	elif [ "$$svd" = trl_one_alway ]; then svd="trlanczos -svd_trlanczos_oneside -bv_orthog_refine always"; fi; \
	${MPIEXEC} -n 1 ./test3 -svd_type $$svd -svd_nsv 4 -mat_type aijcusparse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_4:
	-@${SETTEST}; check=test3_1; \
	${MPIEXEC} -n 2 ./test3 -svd_type lapack -svd_nsv 4 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_1: runtest4_1_lanczos runtest4_1_trlanczos runtest4_1_cross runtest4_1_cross_exp runtest4_1_cyclic runtest4_1_cyclic_imp runtest4_1_cyclic_exp
runtest4_1_%:
	-@${SETTEST}; check=test4_1; svd=$*; \
	if [ "$$svd" = trlanczos ]; then svd="trlanczos -svd_ncv 12"; \
	elif [ "$$svd" = cross_exp ]; then svd="cross -svd_cross_explicitmatrix"; \
	elif [ "$$svd" = cross_exp_imp ]; then svd="cross -svd_cross_explicitmatrix -svd_implicittranspose"; \
	elif [ "$$svd" = cyclic_imp ]; then svd="cyclic -svd_implicittranspose"; \
	elif [ "$$svd" = cyclic_exp ]; then svd="cyclic -svd_cyclic_explicitmatrix"; fi; \
	${MPIEXEC} -n 1 ./test4 -svd_type $$svd -svd_monitor_cancel | ${GREP} -v "Transpose mode" >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_2_cuda: runtest4_2_cuda_lanczos runtest4_2_cuda_trlanczos runtest4_2_cuda_cross
runtest4_2_cuda_%:
	-@${SETTEST}; check=test4_1; svd=$*; \
	if [ "$$svd" = trlanczos ]; then svd="trlanczos -svd_ncv 12"; fi; \
	${MPIEXEC} -n 1 ./test4 -svd_type $$svd -svd_monitor_cancel -mat_type aijcusparse  | ${GREP} -v "Transpose mode" >> $${test}.tmp 2>&1; \
	${TESTCODE_CUDA}

runtest4_3:
	-@${SETTEST}; \
	${MPIEXEC} -n 2 ./test4 -svd_type trlanczos -svd_ncv 12 -svd_monitor_cancel -ds_parallel synchronized >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test4f >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test5 -svd_error_relative ::ascii_info_detail -svd_view_values -svd_monitor_conv -svd_error_absolute ::ascii_matlab -svd_monitor_all -svd_converged_reason -svd_view | ${GREP} -v "tolerance" | ${GREP} -v "problem type" | ${SED} -e "s/[0-9]\.[0-9]*e-[0-9]*/removed/g" >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_1: runtest6_1_subspace runtest6_1_lobpcg
runtest6_1_%:
	-@${SETTEST}; check=test6_1; eps=$*; \
	if [ "$$eps" = lobpcg ]; then eps="lobpcg -st_type precond"; fi; \
	${MPIEXEC} -n 1 ./test6 -eps_type $$eps >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_2_cuda:
	-@${SETTEST}; check=test6_1; \
	${MPIEXEC} -n 1 ./test6 -eps_type subspace -mat_type aijcusparse >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test7 -info_exclude svd -log_exclude svd >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_2_cuda:
	-@${SETTEST}; check=test7_1; \
	${MPIEXEC} -n 1 ./test7 -info_exclude svd -log_exclude svd -mat_type aijcusparse >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_1: runtest8_1_lanczos runtest8_1_trlanczos runtest8_1_cross runtest8_1_cyclic runtest8_1_lapack
runtest8_1_%:
	-@${SETTEST}; check=test8_1; svd=$*; \
	${MPIEXEC} -n 1 ./test8 -svd_type $$svd -svd_nsv 3 >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_1: runtest9_1_lanczos runtest9_1_trlanczos runtest9_1_cross runtest9_1_cyclic runtest9_1_lapack
runtest9_1_%:
	-@${SETTEST}; check=test9_1; svd=$*; \
	${MPIEXEC} -n 1 ./test9 -svd_type $$svd -svd_nsv 3 >> $${test}.tmp 2>&1; \
	${TESTCODE}

