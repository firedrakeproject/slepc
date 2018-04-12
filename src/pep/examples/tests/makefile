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
LOCDIR     = src/pep/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c
EXAMPLESF  = test3f.F
MANSEC     = PEP
TESTS      = test1 test2 test3 test3f test4 test5 test6

TESTEXAMPLES_C                     = test2.PETSc runtest2_2 runtest2_4 runtest2_5 \
                                                 runtest2_8 test2.rm \
                                     test3.PETSc runtest3_1 test3.rm \
                                     test4.PETSc runtest4_1 test4.rm
TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE = test1.PETSc runtest1_1 test1.rm \
                                     test5.PETSc runtest5_1 test5.rm
TESTEXAMPLES_C_COMPLEX             = test2.PETSc runtest2_2_jd runtest2_12 test2.rm
TESTEXAMPLES_C_NOTSINGLE           = test2.PETSc runtest2_1 runtest2_3 runtest2_2_stoar \
                                                 runtest2_6 runtest2_7 runtest2_9 test2.rm \
                                     test6.PETSc runtest6_1 runtest6_2 test6.rm
TESTEXAMPLES_C_DOUBLE              = test2.PETSc runtest2_10 test2.rm
TESTEXAMPLES_FORTRAN               = test3f.PETSc runtest3f_1 test3f.rm
TESTEXAMPLES_VECCUDA               = test2.PETSc runtest2_11_cuda test2.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_PEP_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_PEP_LIB}
	${RM} test2.o

test3: test3.o chkopts
	-${CLINKER} -o test3 test3.o ${SLEPC_PEP_LIB}
	${RM} test3.o

test3f: test3f.o chkopts
	-${FLINKER} -o test3f test3f.o ${SLEPC_PEP_LIB}
	${RM} test3f.o

test4: test4.o chkopts
	-${CLINKER} -o test4 test4.o ${SLEPC_PEP_LIB}
	${RM} test4.o

test5: test5.o chkopts
	-${CLINKER} -o test5 test5.o ${SLEPC_PEP_LIB}
	${RM} test5.o

test6: test6.o chkopts
	-${CLINKER} -o test6 test6.o ${SLEPC_PEP_LIB}
	${RM} test6.o

#------------------------------------------------------------------------------------

runtest1_1: runtest1_1_toar runtest1_1_qarnoldi runtest1_1_linear runtest1_1_linear_gd
runtest1_1_%:
	-@${SETTEST}; check=test1_1; pep=$*; \
	if [ "$$pep" = linear_gd ]; then pep="linear -epstype gd"; fi; \
	${MPIEXEC} -n 1 ./test1 -type $${pep} -m 11 >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1: runtest2_1_toar runtest2_1_qarnoldi runtest2_1_linear runtest2_1_linear_gd
runtest2_1_%:
	-@${SETTEST}; check=test2_1; pep=$*; \
	if [ "$$pep" = linear_gd ]; then pep="linear -pep_linear_eps_type gd -pep_linear_explicitmatrix"; \
	elif [ "$$pep" = qarnoldi ]; then pep="qarnoldi -bv_orthog_refine never"; fi; \
	${MPIEXEC} -n 1 ./test2 -pep_type $${pep} -pep_nev 4 -initv >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_2: runtest2_2_toar runtest2_2_toar_scaleboth runtest2_2_toar_transform runtest2_2_qarnoldi runtest2_2_linear runtest2_2_linear_explicit runtest2_2_linear_explicit_her
runtest2_2_%:
	-@${SETTEST}; check=test2_2; pep=$*; \
	if [ "$$pep" != jd ]; then sinv="-st_type sinvert"; fi; \
	if [ "$$pep" = toar_scaleboth ]; then pep="toar -pep_scale both"; \
	elif [ "$$pep" = toar_transform ]; then pep="toar -st_transform"; \
	elif [ "$$pep" = stoar ]; then pep="stoar -pep_hermitian"; \
	elif [ "$$pep" = qarnoldi ]; then pep="qarnoldi -bv_orthog_refine always"; \
	elif [ "$$pep" = linear_explicit ]; then pep="linear -pep_linear_explicitmatrix -pep_linear_cform 2"; \
	elif [ "$$pep" = linear_explicit_her ]; then pep="linear -pep_linear_explicitmatrix -pep_hermitian -pep_linear_cform 2"; fi; \
	${MPIEXEC} -n 1 ./test2 -pep_type $${pep} -pep_target -0.43 -pep_nev 4 -pep_ncv 20 $$sinv >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_3: runtest2_3_none runtest2_3_norm runtest2_3_residual runtest2_3_structured
runtest2_3_%:
	-@${SETTEST}; check=test2_3; extract=$*; \
	${MPIEXEC} -n 1 ./test2 -pep_nev 12 -pep_extract $${extract} -pep_monitor_cancel >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_4: runtest2_4_schur runtest2_4_mbe runtest2_4_explicit runtest2_4_multiple_schur runtest2_4_multiple_mbe runtest2_4_multiple_explicit
runtest2_4_%:
	-@${SETTEST}; check=test2_2; refine=$*; \
	if [ "$$refine" = schur ]; then refine="simple -pep_refine_scheme schur"; \
	elif [ "$$refine" = mbe ]; then refine="simple -pep_refine_scheme mbe -pep_refine_ksp_type preonly -pep_refine_pc_type lu"; \
	elif [ "$$refine" = explicit ]; then refine="simple -pep_refine_scheme explicit"; \
	elif [ "$$refine" = multiple_schur ]; then refine="multiple -pep_refine_scheme schur"; \
	elif [ "$$refine" = multiple_mbe ]; then refine="multiple -pep_refine_scheme mbe -pep_refine_ksp_type preonly -pep_refine_pc_type lu -pep_refine_pc_factor_shift_type nonzero"; \
	elif [ "$$refine" = multiple_explicit ]; then refine="multiple -pep_refine_scheme explicit"; fi; \
	${MPIEXEC} -n 1 ./test2 -st_type sinvert -pep_target -0.43 -pep_nev 4 -pep_refine $${refine} >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_5:
	-@${SETTEST}; check=test2_2; \
	${MPIEXEC} -n 2 ./test2 -pep_type linear -pep_linear_explicitmatrix -pep_general -pep_target -0.43 -pep_nev 4 -pep_ncv 20 -st_type sinvert -pep_linear_st_ksp_type bcgs -pep_linear_st_pc_type bjacobi >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_6: runtest2_6_none runtest2_6_norm runtest2_6_residual
runtest2_6_%:
	-@${SETTEST}; check=test2_3; extract=$*; \
	${MPIEXEC} -n 1 ./test2 -pep_type linear -pep_nev 12 -pep_extract $${extract} >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_7: runtest2_7_none runtest2_7_norm runtest2_7_residual runtest2_7_structured
runtest2_7_%:
	-@${SETTEST}; check=test2_3; extract=$*; \
	${MPIEXEC} -n 1 ./test2 -pep_nev 12 -pep_extract $${extract} -pep_refine multiple >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_8: runtest2_8_schur runtest2_8_mbe runtest2_8_explicit runtest2_8_multiple_schur runtest2_8_multiple_mbe runtest2_8_multiple_explicit
runtest2_8_%:
	-@${SETTEST}; check=test2_2; refine=$*; \
	if [ "$$refine" = schur ]; then refine="simple -pep_refine_scheme schur"; \
	elif [ "$$refine" = mbe ]; then refine="simple -pep_refine_scheme mbe -pep_refine_ksp_type preonly -pep_refine_pc_type lu"; \
	elif [ "$$refine" = explicit ]; then refine="simple -pep_refine_scheme explicit"; \
	elif [ "$$refine" = multiple_schur ]; then refine="multiple -pep_refine_scheme schur"; \
	elif [ "$$refine" = multiple_mbe ]; then refine="multiple -pep_refine_scheme mbe -pep_refine_ksp_type preonly -pep_refine_pc_type lu"; \
	elif [ "$$refine" = multiple_explicit ]; then refine="multiple -pep_refine_scheme explicit"; fi; \
	${MPIEXEC} -n 1 ./test2 -st_type sinvert -pep_target -0.43 -pep_nev 4 -pep_refine $${refine} -st_transform >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_9: runtest2_9_mbe runtest2_9_explicit runtest2_9_multiple_mbe runtest2_9_multiple_explicit
runtest2_9_%:
	-@${SETTEST}; check=test2_2; refine=$*; \
	if [ "$$refine" = mbe ]; then refine="simple -pep_refine_scheme mbe -pep_refine_ksp_type preonly -pep_refine_pc_type lu"; \
	elif [ "$$refine" = explicit ]; then refine="simple -pep_refine_scheme explicit"; \
	elif [ "$$refine" = multiple_mbe ]; then refine="multiple -pep_refine_scheme mbe -pep_refine_ksp_type preonly -pep_refine_pc_type lu"; \
	elif [ "$$refine" = multiple_explicit ]; then refine="multiple -pep_refine_scheme explicit"; fi; \
	${MPIEXEC} -n 2 ./test2 -st_type sinvert -pep_target -0.43 -pep_nev 4 -pep_refine $${refine} -pep_refine_partitions 2 -st_ksp_type bcgs -st_pc_type bjacobi -pep_scale diagonal -pep_scale_its 4 >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_10:
	-@${SETTEST}; check=test2_2; \
	${MPIEXEC} -n 4 ./test2 -st_type sinvert -pep_target -0.43 -pep_nev 4 -pep_refine simple -pep_refine_scheme explicit -pep_refine_partitions 2 -st_ksp_type bcgs -st_pc_type bjacobi -pep_scale diagonal -pep_scale_its 4 >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_11_cuda: runtest2_11_cuda_toar runtest2_11_cuda_qarnoldi runtest2_11_cuda_linear runtest2_11_cuda_linear_gd
runtest2_11_cuda_%:
	-@${SETTEST}; check=test2_1; pep=$*; \
	if [ "$$pep" = linear_gd ]; then pep="linear -pep_linear_eps_type gd -pep_linear_explicitmatrix"; \
	elif [ "$$pep" = qarnoldi ]; then pep="qarnoldi -bv_orthog_refine never"; fi; \
	${MPIEXEC} -n 1 ./test2 -pep_type $${pep} -pep_nev 4 -initv -mat_type aijcusparse >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_12:
	-@${SETTEST}; \
	${MPIEXEC} -n 2 ./test2 -pep_type jd -ds_parallel synchronized -pep_target -0.43 -pep_nev 4 -pep_ncv 20 >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test3 -pep_tol 1e-6 -pep_ncv 22 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test3f -pep_tol 1e-6 -pep_ncv 22 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test4 -pep_linear_explicitmatrix > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test5 -pep_error_backward ::ascii_info_detail -pep_largest_real -pep_view_values -pep_monitor_conv -pep_error_absolute ::ascii_matlab -pep_monitor_all -pep_converged_reason -pep_view | ${GREP} -v "tolerance" | ${GREP} -v "problem type" | ${SED} -e "s/[0-9]\.[0-9]*e-[0-9]*/removed/g" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_1: runtest6_1_toar runtest6_1_qarnoldi runtest6_1_linear
runtest6_1_%:
	-@${SETTEST}; check=test6_1; pep=$*; \
	${MPIEXEC} -n 1 ./test6 -pep_type $$pep -pep_nev 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_2: 
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test6 -pep_type stoar -pep_hermitian -pep_nev 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}


