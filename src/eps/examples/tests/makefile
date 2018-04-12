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
LOCDIR     = src/eps/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c \
             test8.c test9.c test10.c test11.c test12.c test13.c \
             test14.c test16.c test17.c test18.c test19.c test20.c \
             test21.c test22.c test23.c test24.c test25.c test26.c test27.c \
             test28.c
EXAMPLESF  = test7f.F test14f.F test15f.F test17f.F90
MANSEC     = EPS
TESTS      = test1 test2 test3 test4 test5 test6 test7f test8 test9 test10 \
             test11 test12 test13 test14 test14f test15f test16 test17 test17f \
             test18 test19 test20 test21 test22 test23 test24 test25 test26 test27 \
             test28

TESTEXAMPLES_C                     = test1.PETSc runtest1_5 runtest1_9 test1.rm \
                                     test4.PETSc runtest4_2 test4.rm \
                                     test8.PETSc runtest8_2 runtest8_3 test8.rm \
                                     test13.PETSc runtest13_1 runtest13_2 test13.rm \
                                     test14.PETSc runtest14_1 test14.rm \
                                     test19.PETSc runtest19_1 test19.rm \
                                     test20.PETSc runtest20_1 test20.rm \
                                     test21.PETSc runtest21_1 test21.rm
TESTEXAMPLES_C_NOTSINGLE           = test1.PETSc runtest1_1 runtest1_3 runtest1_4 runtest1_6 test1.rm \
                                     test2.PETSc runtest2_1 runtest2_2 runtest2_3 test2.rm \
                                     test3.PETSc runtest3_1 runtest3_2 test3.rm \
                                     test4.PETSc runtest4_1 test4.rm \
                                     test6.PETSc runtest6_1 test6.rm \
                                     test8.PETSc runtest8_1 test8.rm \
                                     test9.PETSc runtest9_1 runtest9_2 runtest9_3 \
                                                 runtest9_4 runtest9_6 test9.rm \
                                     test10.PETSc runtest10_1 test10.rm \
                                     test11.PETSc runtest11_1 test11.rm \
                                     test12.PETSc runtest12_1 test12.rm \
                                     test16.PETSc runtest16_1 test16.rm \
                                     test17.PETSc runtest17_1 test17.rm \
                                     test18.PETSc runtest18_1 test18.rm \
                                     test24.PETSc runtest24_1 test24.rm \
                                     test27.PETSc runtest27_1 runtest27_2 test27.rm \
                                     test28.PETSc runtest28_1 runtest28_2 runtest28_3 test28.rm
TESTEXAMPLES_C_DATAFILE            = test25.PETSc runtest25_1 test25.rm \
                                     test26.PETSc runtest26_1 test26.rm
TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE = test1.PETSc runtest1_2 test1.rm \
                                     test9.PETSc runtest9_5 test9.rm \
                                     test18.PETSc runtest18_1_gd runtest18_1_jd test18.rm \
                                     test22.PETSc runtest22_1 runtest22_2 test22.rm \
                                     test23.PETSc runtest23_1 test23.rm
TESTEXAMPLES_C_DOUBLE              = test5.PETSc runtest5_1 test5.rm
TESTEXAMPLES_FORTRAN               = test7f.PETSc runtest7f_1 test7f.rm \
                                     test14f.PETSc runtest14f_1 test14f.rm \
                                     test17f.PETSc runtest17f_1 test17f.rm
TESTEXAMPLES_VECCUDA               = test1.PETSc runtest1_7_cuda runtest1_8_cuda test1.rm \
                                     test2.PETSc runtest2_4_cuda test2.rm \
                                     test18.PETSc runtest18_2_cuda test18.rm \
                                     test24.PETSc runtest24_2_cuda test24.rm
TESTEXAMPLES_VECCUDA_NOCOMPLEX     = test25.PETSc runtest25_2_cuda test25.rm
TESTEXAMPLES_ARPACK                = test3.PETSc runtest3_1_arpack test3.rm \
                                     test4.PETSc runtest4_1_arpack test4.rm \
                                     test6.PETSc runtest6_1_arpack test6.rm \
                                     test8.PETSc runtest8_2_arpack test8.rm
TESTEXAMPLES_BLZPACK               = test4.PETSc runtest4_2_blzpack test4.rm \
                                     test5.PETSc runtest5_2_blzpack test5.rm \
                                     test8.PETSc runtest8_2_blzpack test8.rm
TESTEXAMPLES_BLOPEX                = test5.PETSc runtest5_2_blopex test5.rm \
                                     test8.PETSc runtest8_2_blopex test8.rm
TESTEXAMPLES_PRIMME                = test3.PETSc runtest3_1_primme test3.rm \
                                     test4.PETSc runtest4_1_primme test4.rm \
                                     test8.PETSc runtest8_1_primme test8.rm
TESTEXAMPLES_TRLAN                 = test3.PETSc runtest3_1_trlan runtest3_2_trlan test3.rm \
                                     test4.PETSc runtest4_1_trlan test4.rm \
                                     test5.PETSc runtest5_2_trlan test5.rm \
                                     test6.PETSc runtest6_1_trlan test6.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_EPS_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_EPS_LIB}
	${RM} test2.o

test3: test3.o chkopts
	-${CLINKER} -o test3 test3.o ${SLEPC_EPS_LIB}
	${RM} test3.o

test4: test4.o chkopts
	-${CLINKER} -o test4 test4.o ${SLEPC_EPS_LIB}
	${RM} test4.o

test5: test5.o chkopts
	-${CLINKER} -o test5 test5.o ${SLEPC_EPS_LIB}
	${RM} test5.o

test6: test6.o chkopts
	-${CLINKER} -o test6 test6.o ${SLEPC_EPS_LIB}
	${RM} test6.o

test7f: test7f.o chkopts
	-${FLINKER} -o test7f test7f.o ${SLEPC_EPS_LIB}
	${RM} test7f.o

test8: test8.o chkopts
	-${CLINKER} -o test8 test8.o ${SLEPC_EPS_LIB}
	${RM} test8.o

test9: test9.o chkopts
	-${CLINKER} -o test9 test9.o ${SLEPC_EPS_LIB}
	${RM} test9.o

test10: test10.o chkopts
	-${CLINKER} -o test10 test10.o ${SLEPC_EPS_LIB}
	${RM} test10.o

test11: test11.o chkopts
	-${CLINKER} -o test11 test11.o ${SLEPC_EPS_LIB}
	${RM} test11.o

test12: test12.o chkopts
	-${CLINKER} -o test12 test12.o ${SLEPC_EPS_LIB}
	${RM} test12.o

test13: test13.o chkopts
	-${CLINKER} -o test13 test13.o ${SLEPC_EPS_LIB}
	${RM} test13.o

test14: test14.o chkopts
	-${CLINKER} -o test14 test14.o ${SLEPC_EPS_LIB}
	${RM} test14.o

test14f: test14f.o chkopts
	-${FLINKER} -o test14f test14f.o ${SLEPC_EPS_LIB}
	${RM} test14f.o

test15f: test15f.o chkopts
	-${FLINKER} -o test15f test15f.o ${SLEPC_EPS_LIB}
	${RM} test15f.o

test16: test16.o chkopts
	-${CLINKER} -o test16 test16.o ${SLEPC_EPS_LIB}
	${RM} test16.o

test17: test17.o chkopts
	-${CLINKER} -o test17 test17.o ${SLEPC_EPS_LIB}
	${RM} test17.o

test17f: test17f.o chkopts
	-${FLINKER} -o test17f test17f.o ${SLEPC_EPS_LIB}
	${RM} test17f.o

test18: test18.o chkopts
	-${CLINKER} -o test18 test18.o ${SLEPC_EPS_LIB}
	${RM} test18.o

test19: test19.o chkopts
	-${CLINKER} -o test19 test19.o ${SLEPC_EPS_LIB}
	${RM} test19.o

test20: test20.o chkopts
	-${CLINKER} -o test20 test20.o ${SLEPC_EPS_LIB}
	${RM} test20.o

test21: test21.o chkopts
	-${CLINKER} -o test21 test21.o ${SLEPC_EPS_LIB}
	${RM} test21.o

test22: test22.o chkopts
	-${CLINKER} -o test22 test22.o ${SLEPC_EPS_LIB}
	${RM} test22.o

test23: test23.o chkopts
	-${CLINKER} -o test23 test23.o ${SLEPC_EPS_LIB}
	${RM} test23.o

test24: test24.o chkopts
	-${CLINKER} -o test24 test24.o ${SLEPC_EPS_LIB}
	${RM} test24.o

test25: test25.o chkopts
	-${CLINKER} -o test25 test25.o ${SLEPC_EPS_LIB}
	${RM} test25.o

test26: test26.o chkopts
	-${CLINKER} -o test26 test26.o ${SLEPC_EPS_LIB}
	${RM} test26.o

test27: test27.o chkopts
	-${CLINKER} -o test27 test27.o ${SLEPC_EPS_LIB}
	${RM} test27.o

test28: test28.o chkopts
	-${CLINKER} -o test28 test28.o ${SLEPC_EPS_LIB}
	${RM} test28.o

#------------------------------------------------------------------------------------
DATAPATH = ${SLEPC_DIR}/share/slepc/datafiles/matrices

runtest1_1: runtest1_1_krylovschur runtest1_1_ks_nopurify runtest1_1_ks_trueres runtest1_1_ks_sinvert runtest1_1_ks_cayley runtest1_1_subspace runtest1_1_arnoldi runtest1_1_lanczos runtest1_1_gd runtest1_1_jd runtest1_1_gd2 runtest1_1_ciss runtest1_1_lapack runtest1_1_lobpcg
runtest1_1_%:
	-@${SETTEST}; check=test1_1; eps=$*; \
	if [ "$$eps" = ks_nopurify ]; then eps="krylovschur -eps_purify 0"; \
	elif [ "$$eps" = ks_trueres ]; then eps="krylovschur -eps_true_residual"; \
	elif [ "$$eps" = ks_sinvert ]; then eps="krylovschur -st_type sinvert -eps_target 22"; \
	elif [ "$$eps" = ks_cayley ]; then eps="krylovschur -st_type cayley -eps_target 22"; \
	elif [ "$$eps" = lanczos ]; then eps="lanczos -eps_lanczos_reorthog full"; \
	elif [ "$$eps" = gd2 ]; then eps="gd -eps_gd_double_expansion"; \
	elif [ "$$eps" = ciss ]; then eps="ciss -rg_interval_endpoints 20.8,22 -eps_largest_real"; \
	elif [ "$$eps" = lobpcg ]; then eps="lobpcg -st_shift 22 -eps_largest_real"; fi; \
	${MPIEXEC} -n 1 ./test1 -n 18 -eps_type $$eps -eps_nev 4 -eps_max_it 1500 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test1 -eps_interval .1,1.1 -eps_tol 1e-10 -st_type sinvert -st_ksp_type preonly -st_pc_type cholesky > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1_3:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test1 -n 18 -eps_type power -eps_nev 3 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1_4: runtest1_4_constant runtest1_4_rayleigh runtest1_4_wilkinson
runtest1_4_%:
	-@${SETTEST}; check=test1_4; stype=$*; \
	${MPIEXEC} -n 1 ./test1 -n 18 -eps_type power -eps_nev 3 -st_type sinvert -eps_target 1.15 -eps_power_shift_type $$stype > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1_5: runtest1_5_rqcg runtest1_5_lobpcg
runtest1_5_%:
	-@${SETTEST}; check=test1_5; eps=$*; \
	if [ "$$eps" = lobpcg ]; then eps="lobpcg -eps_lobpcg_blocksize 3"; fi; \
	${MPIEXEC} -n 1 ./test1 -n 18 -eps_type $$eps -eps_nev 3 -eps_smallest_real -eps_max_it 500 -st_pc_type icc > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1_6: runtest1_6_krylovschur runtest1_6_subspace runtest1_6_arnoldi runtest1_6_lanczos runtest1_6_gd
runtest1_6_%:
	-@${SETTEST}; check=test1_6; eps=$*; \
	if [ "$$eps" = lanczos ]; then eps="lanczos -eps_lanczos_reorthog full"; fi; \
	${MPIEXEC} -n 1 ./test1 -n 18 -eps_type $$eps -eps_nev 12 -eps_mpd 8 -eps_max_it 3000 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1_7_cuda: runtest1_7_cuda_krylovschur runtest1_7_cuda_ks_sinvert runtest1_7_cuda_subspace runtest1_7_cuda_arnoldi runtest1_7_cuda_lanczos runtest1_7_cuda_gd runtest1_7_cuda_jd runtest1_7_cuda_ciss
runtest1_7_cuda_%:
	-@${SETTEST}; check=test1_1; eps=$*; \
	if [ "$$eps" = ks_sinvert ]; then eps="krylovschur -st_type sinvert -eps_target 22"; \
	elif [ "$$eps" = lanczos ]; then eps="lanczos -eps_lanczos_reorthog full"; \
	elif [ "$$eps" = ciss ]; then eps="ciss -rg_interval_endpoints 20.8,22 -eps_largest_real"; fi; \
	${MPIEXEC} -n 1 ./test1 -n 18 -eps_type $$eps -eps_nev 4 -eps_max_it 1500 -mat_type aijcusparse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1_8_cuda: runtest1_8_cuda_rqcg runtest1_8_cuda_lobpcg
runtest1_8_cuda_%:
	-@${SETTEST}; check=test1_5; eps=$*; \
	if [ "$$eps" = lobpcg ]; then eps="lobpcg -eps_lobpcg_blocksize 3"; fi; \
	${MPIEXEC} -n 1 ./test1 -n 18 -eps_type $$eps -eps_nev 3 -eps_smallest_real -eps_max_it 500 -st_pc_type sor -mat_type aijcusparse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1_9: runtest1_9_ks_ghep runtest1_9_ks_gnhep runtest1_9_ks_ghiep runtest1_9_lobpcg_ghep runtest1_9_jd_gnhep
runtest1_9_%:
	-@${SETTEST}; check=test1_9; eps=$*; \
	if [ "$$eps" = ks_ghep ]; then eps="-eps_gen_hermitian -st_pc_type redundant -st_type sinvert"; \
	elif [ "$$eps" = ks_gnhep ]; then eps="-eps_gen_non_hermitian -st_pc_type redundant -st_type sinvert"; \
	elif [ "$$eps" = ks_ghiep ]; then eps="-eps_gen_indefinite -st_pc_type redundant -st_type sinvert"; \
	elif [ "$$eps" = lobpcg_ghep ]; then eps="-eps_gen_hermitian -eps_type lobpcg -eps_max_it 200 -eps_lobpcg_blocksize 6"; \
	elif [ "$$eps" = jd_gnhep ]; then eps="-eps_gen_non_hermitian -eps_type jd -eps_target 0"; fi; \
	${MPIEXEC} -n 2 ./test1 -n 18 $$eps -eps_nev 7 -eps_ncv 32 -ds_parallel synchronized | ${GREP} -v "orthogonality" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1: runtest2_1_krylovschur runtest2_1_arnoldi runtest2_1_gd runtest2_1_jd runtest2_1_gd2 runtest2_1_lapack
runtest2_1_%:
	-@${SETTEST}; check=test2_1; eps=$*; \
	if [ "$$eps" = gd2 ]; then eps="gd -eps_gd_double_expansion"; fi; \
	${MPIEXEC} -n 1 ./test2 -eps_type $$eps -eps_nev 4 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_2: runtest2_2_local runtest2_2_full runtest2_2_selective runtest2_2_periodic runtest2_2_partial
runtest2_2_%:
	-@${SETTEST}; check=test2_1; reorthog=$*; \
	${MPIEXEC} -n 1 ./test2 -eps_type lanczos -eps_nev 4 -eps_lanczos_reorthog $$reorthog | ${GREP} -v "Lanczos" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_3: runtest2_3_krylovschur runtest2_3_gd runtest2_3_jd runtest2_3_lapack
runtest2_3_%:
	-@${SETTEST}; check=test2_3; eps=$*; \
	if [ "$$eps" = jd ]; then eps="jd -eps_jd_krylov_start -eps_ncv 18"; \
	elif [ "$$eps" = gd ]; then eps="gd -eps_gd_krylov_start"; fi; \
	${MPIEXEC} -n 2 ./test2 -n 32 -eps_type $$eps -eps_nev 4 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_4_cuda: runtest2_4_cuda_krylovschur runtest2_4_cuda_arnoldi runtest2_4_cuda_gd runtest2_4_cuda_jd
runtest2_4_cuda_%:
	-@${SETTEST}; check=test2_1; eps=$*; \
        ${MPIEXEC} -n 1 ./test2 -eps_type $$eps -eps_nev 4 -mat_type aijcusparse > $${test}.tmp 2>&1; \
        ${TESTCODE}

runtest3_1: runtest3_1_krylovschur runtest3_1_power runtest3_1_subspace runtest3_1_arnoldi runtest3_1_lanczos runtest3_1_gd runtest3_1_jd runtest3_1_gd2 runtest3_1_lapack
runtest3_1_%:
	-@${SETTEST}; check=test3_1; eps=$*; \
	if [ "$$eps" = gd2 ]; then eps="gd -eps_gd_double_expansion"; \
	elif [ "$$eps" = jd ]; then eps="jd -eps_jd_initial_size 7"; \
	elif [ "$$eps" = gd ]; then eps="gd -eps_gd_initial_size 7"; \
	elif [ "$$eps" = power ]; then eps="power -eps_max_it 20000"; \
	elif [ "$$eps" = primme ]; then eps="primme -eps_conv_abs"; fi; \
	${MPIEXEC} -n 1 ./test3 -eps_type $$eps -eps_nev 4 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_2: runtest3_2_rqcg runtest3_2_lobpcg runtest3_2_lanczos runtest3_2_lanczos_delayed
runtest3_2_%:
	-@${SETTEST}; check=test3_2; eps=$*; \
	if [ "$$eps" = lobpcg ]; then eps="lobpcg -eps_lobpcg_blocksize 6 -st_ksp_type preonly"; \
	elif [ "$$eps" = rqcg ]; then eps="rqcg -eps_rqcg_reset 5 -eps_ncv 32 -st_pc_type bjacobi"; \
	elif [ "$$eps" = lanczos_delayed ]; then eps="lanczos -eps_lanczos_reorthog delayed -eps_tol 1e-8"; fi; \
	${MPIEXEC} -n 1 ./test3 -eps_type $$eps -eps_nev 4 -eps_smallest_real -eps_max_it 500 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_1: runtest4_1_krylovschur runtest4_1_subspace runtest4_1_arnoldi runtest4_1_lanczos runtest4_1_gd runtest4_1_jd runtest4_1_gd2 runtest4_1_lapack
runtest4_1_%:
	-@${SETTEST}; check=test4_1; eps=$*; \
	${MPIEXEC} -n 1 ./test4 -type $$eps > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_2: runtest4_2_rqcg runtest4_2_lobpcg
runtest4_2_%:
	-@${SETTEST}; check=test4_2; eps=$*; \
	${MPIEXEC} -n 1 ./test4 -type $$eps > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_1: runtest5_1_krylovschur runtest5_1_power runtest5_1_subspace runtest5_1_arnoldi runtest5_1_gd runtest5_1_jd runtest5_1_gd2 runtest5_1_lapack
runtest5_1_%:
	-@${SETTEST}; check=test5_1; eps=$*; \
	if [ "${PETSC_SCALAR}" = "complex" ]; then check=$${check}_complex; fi; \
	if [ "$$eps" = jd ]; then eps="jd -eps_jd_minv 3 -eps_jd_plusk 1"; \
	elif [ "$$eps" = gd ]; then eps="gd -eps_gd_minv 3 -eps_gd_plusk 1"; \
	elif [ "$$eps" = gd2 ]; then eps="gd -eps_gd_double_expansion"; \
	elif [ "$$eps" = power ]; then eps="power -st_type sinvert -eps_target 7"; fi; \
	${MPIEXEC} -n 1 ./test5 -eps_type $$eps -eps_nev 3 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_2_%:
	-@${SETTEST}; check=test5_2; eps=$*; \
	if [ "${PETSC_SCALAR}" = "complex" ]; then check=$${check}_complex; fi; \
	${MPIEXEC} -n 1 ./test5 -symm -eps_type $$eps -eps_nev 4 -eps_smallest_real > $${test}.tmp 2>&1; \
	${TESTCODE}

testtest5_blopex: test5.PETSc
	@fail=0; if [ "${PETSC_WITH_BATCH}" != "" -o "${MPIEXEC}" = "/bin/false" ]; then \
	  echo "Skipping BLOPEX test"; \
	elif [ -f test5 ]; then \
	  test=test5_2_blopex; check=test5_2; \
	  if [ "${PETSC_SCALAR}" = "complex" ]; then check=$${check}_complex; fi; \
	  ${MPIEXEC} -n 1 ./test5 -symm -eps_type blopex -eps_nev 4 > $${test}.tmp 2>&1; \
	  if (${DIFF} output/$${check}.out $${test}.tmp > /dev/null 2>&1) then \
	    echo "BLOPEX example src/eps/examples/tests/test5 run successfully with 1 MPI process"; \
	  else \
	    echo "Possible error running BLOPEX src/eps/examples/tests/test5 with 1 MPI process"; \
	    cat $${test}.tmp; fail=1; \
	  fi; \
	  ${RM} $${test}.tmp; \
	  ${MAKE} SLEPC_ARCH=${SLEPC_ARCH} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} test5.rm ; \
	else fail=1; fi; \
	exit $$fail

runtest6_1: runtest6_1_krylovschur runtest6_1_power runtest6_1_subspace runtest6_1_arnoldi runtest6_1_gd runtest6_1_jd runtest6_1_gd2
runtest6_1_%:
	-@${SETTEST}; check=test6_1; eps=$*; \
	if [ "$$eps" = gd2 ]; then eps="gd -eps_gd_double_expansion"; \
	elif [ "$$eps" = power ]; then eps="power -eps_max_it 20000"; fi; \
	${MPIEXEC} -n 1 ./test6 -eps_type $$eps -eps_nev 4 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test7f -eps_nev 4 > $${test}.tmp 2>&1; \
	${TESTCODE}

testtest7f: test7f.PETSc
	@fail=0; if [ "${PETSC_WITH_BATCH}" != "" ]; then \
	  echo "Running with batch filesystem; to test run src/eps/examples/tests/test7f " ; \
	  echo "with your systems batch system"; \
	elif [ "${MPIEXEC}" = "/bin/false" ]; then \
	  echo "*mpiexec not found*. Please run src/eps/examples/tests/test7f manually"; \
	elif [ -f test7f ]; then \
	  test=test7f_1; check=test7f_1; \
	  GFORTRAN_UNBUFFERED_ALL=1 ${MPIEXEC} -n 1 ./test7f -eps_nev 4 > $${test}.tmp 2>&1; \
	  if (${DIFF} output/$${check}.out $${test}.tmp > /dev/null 2>&1) then \
	    echo "Fortran example src/eps/examples/tests/test7f run successfully with 1 MPI process"; \
	  else \
	    echo "Possible error running Fortran src/eps/examples/tests/test7f with 1 MPI process"; \
	    cat $${test}.tmp; fail=1; \
	  fi; \
	  ${RM} $${test}.tmp; \
	  ${MAKE} SLEPC_ARCH=${SLEPC_ARCH} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} test7f.rm ; \
	else fail=1; fi; \
	exit $$fail

runtest8_1: runtest8_1_krylovschur runtest8_1_krylovschur_vecs runtest8_1_power runtest8_1_subspace runtest8_1_arnoldi runtest8_1_lanczos runtest8_1_gd runtest8_1_jd runtest8_1_gd2 runtest8_1_lapack
runtest8_1_%:
	-@${SETTEST}; check=test8_1; eps=$*; \
	if [ "$$eps" = krylovschur_vecs ]; then eps="krylovschur -bv_type vecs -bv_orthog_refine always"; \
	elif [ "$$eps" = gd2 ]; then eps="gd -eps_gd_double_expansion"; \
	elif [ "$$eps" = jd ]; then eps="jd -eps_jd_blocksize 3"; \
	elif [ "$$eps" = gd ]; then eps="gd -eps_gd_blocksize 3"; \
	elif [ "$$eps" = primme ]; then eps="primme -eps_conv_abs"; fi; \
	${MPIEXEC} -n 1 ./test8 -n 20 -eps_type $$eps -eps_nev 4 -eps_ncv 11 -eps_max_it 40000 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_2: runtest8_2_rqcg runtest8_2_lobpcg runtest8_2_lanczos
runtest8_2_%:
	-@${SETTEST}; check=test8_2; eps=$*; \
	if [ "$$eps" = arpack ]; then eps="arpack -eps_ncv 6"; fi; \
	if [ "${PETSC_PRECISION}" = "single" ]; then EXTRA="-eps_tol 1e-5"; fi; \
	${MPIEXEC} -n 1 ./test8 -eps_type $$eps -eps_nev 4 -eps_smallest_real -eps_max_it 500 $$EXTRA > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_3: runtest8_3_rqcg runtest8_3_lobpcg runtest8_3_lanczos
runtest8_3_%:
	-@${SETTEST}; check=test8_3; eps=$*; \
	if [ "${PETSC_PRECISION}" = "single" ]; then EXTRA="-eps_tol 1e-5"; \
	elif [ "${PETSC_PRECISION}" = "__float128" ]; then EXTRA="-eps_tol 1e-25"; fi; \
	if [ "$$eps" = lobpcg ]; then eps="lobpcg -eps_lobpcg_blocksize 3 -eps_lobpcg_locking 0 -st_ksp_type preonly -st_pc_type jacobi"; fi; \
	${MPIEXEC} -n 1 ./test8 -eps_type $$eps -eps_nev 12 -eps_mpd 9 -eps_smallest_real -eps_max_it 1000 $$EXTRA > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_1: runtest9_1_krylovschur runtest9_1_arnoldi runtest9_1_gd runtest9_1_gd2 runtest9_1_lapack
runtest9_1_%:
	-@${SETTEST}; check=test9_1; eps=$*; \
	if [ "$$eps" = "krylovschur" -o "$$eps" = "arnoldi" ]; then EXTRA="-eps_ncv 7 -eps_max_it 300"; \
	elif [ "$$eps" = gd2 ]; then eps="gd -eps_gd_double_expansion"; fi; \
	${MPIEXEC} -n 1 ./test9 -eps_type $$eps -eps_nev 4 $$EXTRA > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_2: runtest9_2_none runtest9_2_oneside runtest9_2_twoside
runtest9_2_%:
	-@${SETTEST}; check=test9_1; balance=$*; \
	${MPIEXEC} -n 1 ./test9 -eps_balance $$balance -eps_nev 4 -eps_ncv 7 -eps_max_it 500 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_3: runtest9_3_never runtest9_3_ifneeded
runtest9_3_%:
	-@${SETTEST}; check=test9_3; refine=$*; \
	${MPIEXEC} -n 2 ./test9 -eps_type arnoldi -eps_arnoldi_delayed -eps_largest_real -eps_nev 3 -eps_tol 1e-7 -bv_orthog_refine $$refine > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_4:
	-@${SETTEST}; check=test9_1; \
	${MPIEXEC} -n 1 ./test9 -eps_nev 4 -eps_true_residual > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_5:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test9 -eps_type jd -eps_nev 3 -eps_target .5 -eps_harmonic -st_ksp_type bicg -eps_jd_minv 2 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_6: runtest9_6_ciss runtest9_6_ciss_hankel runtest9_6_ciss_cheby runtest9_6_ciss_hankel_cheby runtest9_6_ciss_refine runtest9_6_ciss_bcgs
runtest9_6_%:
	-@${SETTEST};  check=test9_6; eps=$*; \
	if [ "$$eps" = ciss_hankel ]; then eps="ciss -eps_ciss_extraction hankel -eps_ciss_spurious_threshold 1e-6"; \
	elif [ "$$eps" = ciss_cheby ]; then eps="ciss -eps_ciss_quadrule chebyshev"; \
	elif [ "$$eps" = ciss_hankel_cheby ]; then eps="ciss -eps_ciss_extraction hankel -eps_ciss_quadrule chebyshev"; \
	elif [ "$$eps" = ciss_refine ]; then eps="ciss -eps_ciss_refine_inner 1 -eps_ciss_refine_blocksize 1"; \
	elif [ "$$eps" = ciss_bcgs ]; then eps="ciss -eps_ciss_realmats -eps_ciss_ksp_type bcgs -eps_ciss_pc_type sor"; fi; \
	${MPIEXEC} -n 1 ./test9 -eps_type $$eps -eps_tol 1e-9 -rg_type ellipse -rg_ellipse_center 0.55 -rg_ellipse_radius 0.05 -rg_ellipse_vscale 0.1 -eps_ciss_usest 0 -eps_all > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest10_1: runtest10_1_krylovschur runtest10_1_arnoldi runtest10_1_lanczos runtest10_1_gd runtest10_1_jd runtest10_1_gd2 runtest10_1_rqcg runtest10_1_lobpcg
runtest10_1_%:
	-@${SETTEST}; check=test10_1; eps=$*; \
	if [ "$$eps" = gd2 ]; then eps="gd -eps_gd_double_expansion"; fi; \
	${MPIEXEC} -n 1 ./test10 -eps_type $$eps -eps_nev 4 -m 11 -eps_max_it 500 > $${test}.tmp 2>&1; \
	${TESTCODE}

testtest10: test10.PETSc
	@fail=0; if [ "${PETSC_WITH_BATCH}" != "" ]; then \
	  echo "Running with batch filesystem; to test run src/eps/examples/tests/test10" ; \
	  echo "with your systems batch system"; \
	elif [ "${MPIEXEC}" = "/bin/false" ]; then \
	  echo "*mpiexec not found*. Please run src/eps/examples/tests/test10 manually"; \
	elif [ -f test10 ]; then \
	  test=test10_1; check=test10_1_ks; \
	  options="-eps_nev 4 -eps_ncv 14 -m 11 -eps_largest_magnitude"; \
	  ${MPIEXEC} -n 1 ./test10 $$options > $${test}.tmp 2>&1; \
	  if (${DIFF} output/$${check}.out $${test}.tmp > /dev/null 2>&1) then \
	    echo "C/C++ example src/eps/examples/tests/test10 run successfully with 1 MPI process"; \
	  else \
	    echo "Possible error running C/C++ src/eps/examples/tests/test10 with 1 MPI process"; \
	    cat $${test}.tmp; fail=1; \
	  fi; \
	  if [ "${MPIEXEC}" != "${PETSC_DIR}/lib/petsc/bin/petsc-mpiexec.uni" ]; then \
	    ${MPIEXEC} -n 2 ./test10 $$options > $${test}.tmp 2>&1; \
	    if (${DIFF} output/$${check}.out $${test}.tmp > /dev/null 2>&1) then \
	      echo "C/C++ example src/eps/examples/tests/test10 run successfully with 2 MPI process"; \
	    else \
	      echo "Possible error running C/C++ src/eps/examples/tests/test10 with 2 MPI process"; \
	      cat $${test}.tmp; fail=1; \
	    fi; \
	  fi; \
	  ${RM} $${test}.tmp; \
	  ${MAKE} SLEPC_ARCH=${SLEPC_ARCH} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} test10.rm; \
	else fail=1; fi; \
	exit $$fail

runtest11_1: runtest11_1_krylovschur runtest11_1_ks_cayley runtest11_1_arnoldi runtest11_1_lapack runtest11_1_gd
runtest11_1_%:
	-@${SETTEST}; check=test11_1; eps=$*; \
	if [ "$$eps" = krylovschur -o "$$eps" = arnoldi -o "$$eps" = lapack ]; then EXTRA="-st_type sinvert"; \
	elif [ "$$eps" = ks_cayley ]; then eps="krylovschur" EXTRA="-st_type cayley -st_cayley_antishift 1"; \
	elif [ "$$eps" = gd ]; then EXTRA="-eps_max_it 5000"; fi; \
	${MPIEXEC} -n 1 ./test11 -eps_type $$eps -eps_nev 4 $$EXTRA > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest12_1: runtest12_1_krylovschur runtest12_1_power runtest12_1_subspace runtest12_1_arnoldi runtest12_1_gd runtest12_1_jd runtest12_1_gd2
runtest12_1_%:
	-@${SETTEST}; check=test12_1; eps=$*; \
	if [ "$$eps" = gd2 ]; then eps="gd -eps_gd_double_expansion"; \
	elif [ "$$eps" = power ]; then eps="power -eps_max_it 10000"; fi; \
	${MPIEXEC} -n 1 ./test12 -eps_type $$eps -eps_nev 4 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest13_1: runtest13_1_krylovschur runtest13_1_gd runtest13_1_jd runtest13_1_gd2
runtest13_1_%:
	-@${SETTEST}; check=test13_1; eps=$*; \
	if [ "$$eps" = jd ]; then eps="jd"; \
	elif [ "$$eps" = gd2 ]; then eps="gd -eps_gd_double_expansion"; fi; \
	${MPIEXEC} -n 1 ./test13 -eps_type $$eps -eps_max_it 5000 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest13_2: runtest13_2_krylovschur runtest13_2_gd runtest13_2_jd runtest13_2_gd2
runtest13_2_%:
	-@${SETTEST}; check=test13_1; eps=$*; \
	if [ "$$eps" = jd ]; then eps="jd"; \
	elif [ "$$eps" = gd2 ]; then eps="gd -eps_gd_double_expansion"; fi; \
	${MPIEXEC} -n 1 ./test13 -eps_type $$eps -eps_non_hermitian -eps_max_it 5000 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest14_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test14 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest14f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test14f > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest15f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test15f -my_eps_monitor > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest16_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test16 -n 200 -eps_nev 6 -eps_ncv 24 -eps_smallest_magnitude > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest17_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 2 ./test17 -showinertia 0 -info_exclude eps,st,rg,bv,ds -log_exclude eps,st,rg,bv,ds > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest17f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 2 ./test17f > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest18_1: runtest18_1_krylovschur runtest18_1_krylovschur_gnhep
runtest18_1_%:
	-@${SETTEST}; check=test18_1; eps=$*; \
	if [ "$$eps" = krylovschur ]; then eps="krylovschur -st_type sinvert -eps_krylovschur_restart .3"; \
	elif [ "$$eps" = krylovschur_gnhep ]; then eps="krylovschur -eps_gen_non_hermitian -st_type sinvert -eps_krylovschur_restart .3"; \
	elif [ "$$eps" = gd ]; then eps="gd -eps_target 0 -eps_harmonic"; \
	elif [ "$$eps" = jd ]; then eps="jd -eps_target 0 -eps_harmonic"; fi; \
	${MPIEXEC} -n 1 ./test18 -eps_type $$eps -eps_nev 4 -eps_ncv 12 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest18_2_cuda: runtest18_2_cuda_krylovschur runtest18_2_cuda_ks_gnhep
runtest18_2_cuda_%:
	-@${SETTEST}; check=test18_1; eps=$*; \
	if [ "$$eps" = krylovschur ]; then eps="krylovschur -st_type sinvert -eps_krylovschur_restart .3"; \
	elif [ "$$eps" = ks_gnhep ]; then eps="krylovschur -eps_gen_non_hermitian -st_type sinvert -eps_krylovschur_restart .3"; fi; \
	${MPIEXEC} -n 1 ./test18 -eps_type $$eps -eps_nev 4 -eps_ncv 12 -terse -mat_type aijcusparse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest19_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test19 -check_myprefix_eps_nev 2 -check_myprefix_st_type sinvert > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest20_1: runtest20_1_krylovschur runtest20_1_arnoldi runtest20_1_lanczos runtest20_1_gd runtest20_1_jd runtest20_1_rqcg runtest20_1_lobpcg runtest20_1_lapack
runtest20_1_%:
	-@${SETTEST}; check=test20_1; eps=$*; \
	if [ "$$eps" = lanczos ]; then eps="lanczos -eps_lanczos_reorthog full"; fi; \
	${MPIEXEC} -n 1 ./test20 -n 18 -eps_type $$eps -eps_max_it 1500 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest21_1:
	-@${SETTEST}; check=test11_1; \
	${MPIEXEC} -n 1 ./test21 -eps_nev 4 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest22_1: runtest22_1_0 runtest22_1_1
runtest22_1_%:
	-@${SETTEST}; check=test22_1; trueres=$*; \
	${MPIEXEC} -n 1 ./test22 -eps_nev 4 -eps_true_residual $$trueres > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest22_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test22 -eps_nev 4 -eps_true_residual -eps_balance oneside -eps_tol 1e-7 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest23_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test23 -eps_error_backward ::ascii_info_detail -eps_largest_real -eps_view_values -eps_monitor_conv -eps_error_absolute ::ascii_matlab -eps_monitor_all -eps_converged_reason -eps_view | ${GREP} -v "tolerance" | ${SED} -e "s/[0-9]\.[0-9]*e-[0-9]*/removed/g" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest24_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test24 -ncon 2 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest24_2_cuda:
	-@${SETTEST}; check=test24_1; \
	${MPIEXEC} -n 1 ./test24 -ncon 2 -mat_type aijcusparse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest25_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test25 -f1 ${DATAPATH}/bfw62a.petsc -f2 ${DATAPATH}/bfw62b.petsc > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest25_2_cuda:
	-@${SETTEST}; check=test25_1; \
	${MPIEXEC} -n 1 ./test25 -f1 ${DATAPATH}/bfw62a.petsc -f2 ${DATAPATH}/bfw62b.petsc -mat_type aijcusparse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest26_1: runtest26_1_0 runtest26_1_1
runtest26_1_%:
	-@${SETTEST}; check=test26_1; trueres=$*; \
	${MPIEXEC} -n 1 ./test26 -f1 ${DATAPATH}/bfw62a.petsc -f2 ${DATAPATH}/bfw62b.petsc -eps_largest_real -eps_nev 4 -eps_true_residual $$trueres > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest27_1: runtest27_1_gd runtest27_1_jd runtest27_1_lobpcg runtest27_1_rqcg
runtest27_1_%:
	-@${SETTEST}; check=test27_1; eps=$*; \
	${MPIEXEC} -n 1 ./test27 -eps_type $$eps -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest27_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test27 -eps_interval .17,1.3 -st_type filter -eps_nev 1 -terse | $(GREP) -v "requested" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest28_1: runtest28_1_krylovschur runtest28_1_arnoldi runtest28_1_lanczos runtest28_1_gd runtest28_1_jd runtest28_1_rqcg runtest28_1_lobpcg runtest28_1_lapack
runtest28_1_%:
	-@${SETTEST}; check=test28_1; eps=$*; \
	${MPIEXEC} -n 1 ./test28 -eps_type $$eps -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest28_2: runtest28_2_power runtest28_2_subspace
runtest28_2_%:
	-@${SETTEST}; check=test28_2; eps=$*; \
	${MPIEXEC} -n 1 ./test28 -eps_type $$eps -eps_target 8 -st_type sinvert -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest28_3:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test28 -eps_interval 0.5,0.67 -st_type sinvert -st_pc_type cholesky -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

