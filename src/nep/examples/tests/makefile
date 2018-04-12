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
LOCDIR     = src/nep/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c test7.c test8.c test9.c test10.c test11.c
EXAMPLESF  = test2f.F
MANSEC     = NEP
TESTS      = test1 test2 test2f test3 test4 test5 test6 test7 test8 test9 test10 test11

TESTEXAMPLES_C_NOTSINGLE       = test1.PETSc runtest1_1 test1.rm \
                                 test2.PETSc runtest2_1 test2.rm \
                                 test3.PETSc runtest3_1 test3.rm \
                                 test4.PETSc runtest4_1 test4.rm \
                                 test5.PETSc runtest5_1 test5.rm \
                                 test7.PETSc runtest7_1 runtest7_2 test7.rm \
                                 test9.PETSc runtest9_1 test9.rm \
                                 test10.PETSc runtest10_1 test10.rm
TESTEXAMPLES_C_COMPLEX         = test10.PETSc runtest10_2 test10.rm \
                                 test11.PETSc runtest11_1 test11.rm
TESTEXAMPLES_C_DATAFILE        = test6.PETSc runtest6_1 test6.rm \
                                 test8.PETSc runtest8_1 test8.rm
TESTEXAMPLES_FORTRAN_NOTSINGLE = test2f.PETSc runtest2f_1 test2f.rm
TESTEXAMPLES_VECCUDA           = test1.PETSc runtest1_2_cuda test1.rm \
                                 test5.PETSc runtest5_2_cuda test5.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_NEP_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_NEP_LIB}
	${RM} test2.o

test2f: test2f.o chkopts
	-${FLINKER} -o test2f test2f.o ${SLEPC_NEP_LIB}
	${RM} test2f.o

test3: test3.o chkopts
	-${CLINKER} -o test3 test3.o ${SLEPC_NEP_LIB}
	${RM} test3.o

test4: test4.o chkopts
	-${CLINKER} -o test4 test4.o ${SLEPC_NEP_LIB}
	${RM} test4.o

test5: test5.o chkopts
	-${CLINKER} -o test5 test5.o ${SLEPC_NEP_LIB}
	${RM} test5.o

test6: test6.o chkopts
	-${CLINKER} -o test6 test6.o ${SLEPC_NEP_LIB}
	${RM} test6.o

test7: test7.o chkopts
	-${CLINKER} -o test7 test7.o ${SLEPC_NEP_LIB}
	${RM} test7.o

test8: test8.o chkopts
	-${CLINKER} -o test8 test8.o ${SLEPC_NEP_LIB}
	${RM} test8.o

test9: test9.o chkopts
	-${CLINKER} -o test9 test9.o ${SLEPC_NEP_LIB}
	${RM} test9.o

test10: test10.o chkopts
	-${CLINKER} -o test10 test10.o ${SLEPC_NEP_LIB}
	${RM} test10.o

test11: test11.o chkopts
	-${CLINKER} -o test11 test11.o ${SLEPC_NEP_LIB}
	${RM} test11.o

#------------------------------------------------------------------------------------

runtest1_1: runtest1_1_rii runtest1_1_slp
runtest1_1_%:
	-@${SETTEST}; check=test1_1; nep=$*; \
	${MPIEXEC} -n 1 ./test1 -nep_type $${nep} -nep_target 21 -terse >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1_2_cuda: runtest1_2_cuda_rii runtest1_2_cuda_slp
runtest1_2_cuda_%:
	-@${SETTEST}; check=test1_1; nep=$*; \
	${MPIEXEC} -n 1 ./test1 -nep_type $${nep} -nep_target 21 -mat_type aijcusparse -terse >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test2 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test2f > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_1:
	-@${SETTEST}; check=test1_1; \
	${MPIEXEC} -n 1 ./test3 -nep_target 21 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest4_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test4 -nep_target 21 -nep_rii_lag_preconditioner 2 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test5 -nep_nev 3 -nep_target 5 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_2_cuda:
	-@${SETTEST}; check=test5_1; \
	${MPIEXEC} -n 1 ./test5 -nep_nev 3 -nep_target 5 -mat_type aijcusparse -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test6 -myprefix_nep_view -myprefix_nep_monitor_conv -initv -terse | ${GREP} -v "tolerance" | ${SED} -e "s/[0-9]\.[0-9]*e[+-][0-9]*/removed/g" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test7 -nep_nev 3 -nep_tol 1e-8 -nep_nleigs_locking 0 -nep_nleigs_interpolation_degree 90 -nep_nleigs_interpolation_tol 1e-8 -nep_nleigs_restart 0.4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test7 -split 0 -nep_nev 3 -nep_tol 1e-8 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test8 -nep_type slp -nep_target -.5 -nep_error_backward ::ascii_info_detail -nep_view_values -nep_error_absolute ::ascii_matlab -nep_monitor_all -nep_converged_reason -nep_view | ${GREP} -v "tolerance" | ${GREP} -v "problem type" | ${SED} -e "s/[0-9]\.[0-9]*e[+-][0-9]*/removed/g" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test9 -nep_type nleigs -rg_type interval -rg_interval_endpoints 4,700,-.1,.1 -nep_nev 8 -nep_target 5 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest10_1: runtest10_1_rii runtest10_1_slp runtest10_1_narnoldi runtest10_1_interpol runtest10_1_narnoldi_sync
runtest10_1_%:
	-@${SETTEST}; check=test10_1; nep=$*; \
	if [ "$$nep" = interpol ]; then nep="interpol -rg_type interval -rg_interval_endpoints .5,1,-.1,.1 -nep_target .7 -nep_interpol_st_pc_type redundant"; \
	elif [ "$$nep" = narnoldi_sync ]; then nep="narnoldi -ds_parallel synchronized"; \
	elif [ "$$nep" = slp ]; then nep="slp -nep_slp_st_pc_type redundant"; fi; \
	${MPIEXEC} -n 2 ./test10 -nep_type $${nep} > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest10_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test10 -nep_nev 2 -nep_type interpol -rg_type interval -rg_interval_endpoints .5,15,-.1,.1 -nep_target .7 -nep_interpol_pep_type jd -nep_interpol_st_pc_type sor > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test11 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

