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
LOCDIR     = src/eps/examples/tutorials/
EXAMPLESC  = ex1.c ex2.c ex3.c ex4.c ex5.c ex7.c ex9.c ex10.c ex11.c ex12.c ex13.c \
             ex18.c ex19.c ex24.c ex25.c ex29.c ex30.c ex31.c ex34.c ex35.c ex36.c
EXAMPLESF  = ex1f.F ex1f90.F90 ex6f90.F90
MANSEC     = EPS

TESTEXAMPLES_C                     = ex5.PETSc runex5_1 ex5.rm \
                                     ex13.PETSc runex13_1 runex13_1 ex13.rm \
                                     ex24.PETSc runex24_1 ex24.rm \
                                     ex29.PETSc runex29_1 ex29.rm \
                                     ex34.PETSc runex34_1 runex34_2 ex34.rm \
                                     ex35.PETSc runex35_1 ex35.rm
TESTEXAMPLES_C_DOUBLE              = ex2.PETSc runex2_ciss_1 runex2_ciss_2 ex2.rm \
                                     ex19.PETSc runex19_1 ex19.rm
TESTEXAMPLES_C_NOTSINGLE           = ex2.PETSc runex2_1 runex2_2 ex2.rm \
                                     ex3.PETSc runex3_1 ex3.rm \
                                     ex10.PETSc runex10_1 ex10.rm \
                                     ex11.PETSc runex11_1 runex11_2 ex11.rm \
                                     ex12.PETSc runex12_1 ex12.rm \
                                     ex18.PETSc runex18_1 ex18.rm \
                                     ex30.PETSc runex30_1 ex30.rm \
                                     ex36.PETSc runex36_1 ex36.rm
TESTEXAMPLES_C_DATAFILE            = ex4.PETSc runex4_1 ex4.rm \
                                     ex7.PETSc runex7_1 ex7.rm \
                                     ex9.PETSc runex9_3 runex9_7 ex9.rm \
                                     ex31.PETSc runex31_1 ex31.rm
TESTEXAMPLES_C_COMPLEX             = ex9.PETSc runex9_6 ex9.rm
TESTEXAMPLES_C_NOCOMPLEX           = ex25.PETSc runex25_1 ex25.rm
TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE = ex9.PETSc runex9_1 runex9_2 runex9_4 runex9_5 runex9_8 ex9.rm
TESTEXAMPLES_FORTRAN_NOCOMPLEX     = ex6f90.PETSc runex6f90_1 ex6f90.rm
TESTEXAMPLES_F90                   = ex1f90.PETSc runex1f90_1 ex1f90.rm
TESTEXAMPLES_DATAFILESPATH_COMPLEX = ex4.PETSc runex4_ciss_1 ex4.rm \
                                     ex7.PETSc runex7_ciss_1 ex7.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${SLEPC_EPS_LIB}
	${RM} ex1.o

ex1f: ex1f.o chkopts
	-${FLINKER} -o ex1f ex1f.o ${SLEPC_EPS_LIB}
	${RM} ex1f.o

ex1f90: ex1f90.o chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ${SLEPC_EPS_LIB}
	${RM} ex1f90.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o ${SLEPC_EPS_LIB}
	${RM} ex2.o

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o ${SLEPC_EPS_LIB}
	${RM} ex3.o

ex4: ex4.o chkopts
	-${CLINKER} -o ex4 ex4.o ${SLEPC_EPS_LIB}
	${RM} ex4.o

ex5: ex5.o chkopts
	-${CLINKER} -o ex5 ex5.o ${SLEPC_EPS_LIB}
	${RM} ex5.o

ex6f90: ex6f90.o chkopts
	-${FLINKER} -o ex6f90 ex6f90.o ${SLEPC_EPS_LIB}
	${RM} ex6f90.o

ex7: ex7.o chkopts
	-${CLINKER} -o ex7 ex7.o ${SLEPC_EPS_LIB}
	${RM} ex7.o

ex9: ex9.o chkopts
	-${CLINKER} -o ex9 ex9.o ${SLEPC_EPS_LIB}
	${RM} ex9.o

ex10: ex10.o chkopts
	-${CLINKER} -o ex10 ex10.o ${SLEPC_EPS_LIB}
	${RM} ex10.o

ex11: ex11.o chkopts
	-${CLINKER} -o ex11 ex11.o ${SLEPC_EPS_LIB}
	${RM} ex11.o

ex12: ex12.o chkopts
	-${CLINKER} -o ex12 ex12.o ${SLEPC_EPS_LIB}
	${RM} ex12.o

ex13: ex13.o chkopts
	-${CLINKER} -o ex13 ex13.o ${SLEPC_EPS_LIB}
	${RM} ex13.o

ex18: ex18.o chkopts
	-${CLINKER} -o ex18 ex18.o ${SLEPC_EPS_LIB}
	${RM} ex18.o

ex19: ex19.o chkopts
	-${CLINKER} -o ex19 ex19.o ${SLEPC_EPS_LIB}
	${RM} ex19.o

ex24: ex24.o chkopts
	-${CLINKER} -o ex24 ex24.o ${SLEPC_EPS_LIB}
	${RM} ex24.o

ex25: ex25.o chkopts
	-${CLINKER} -o ex25 ex25.o ${SLEPC_EPS_LIB}
	${RM} ex25.o

ex29: ex29.o chkopts
	-${CLINKER} -o ex29 ex29.o ${SLEPC_EPS_LIB}
	${RM} ex29.o

ex30: ex30.o chkopts
	-${CLINKER} -o ex30 ex30.o ${SLEPC_EPS_LIB}
	${RM} ex30.o

ex31: ex31.o chkopts
	-${CLINKER} -o ex31 ex31.o ${SLEPC_EPS_LIB}
	${RM} ex31.o

ex34: ex34.o chkopts
	-${CLINKER} -o ex34 ex34.o ${SLEPC_EPS_LIB}
	${RM} ex34.o

ex35: ex35.o chkopts
	-${CLINKER} -o ex35 ex35.o ${SLEPC_EPS_LIB}
	${RM} ex35.o

ex36: ex36.o chkopts
	-${CLINKER} -o ex36 ex36.o ${SLEPC_EPS_LIB} ${SLEPC_MFN_LIB}
	${RM} ex36.o

#------------------------------------------------------------------------------------
DATAPATH = ${SLEPC_DIR}/share/slepc/datafiles/matrices

runex1_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex1 > $${test}.tmp 2>&1; \
	${TESTCODE}

runex1f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex1f > $${test}.tmp 2>&1; \
	${TESTCODE}

runex1f90_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex1f90 -eps_nev 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex2_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex2 -n 72 -eps_nev 4 -eps_ncv 20 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex2_2:
	-@${SETTEST}; check=ex2_1; \
	${MPIEXEC} -n 1 ./ex2 -n 72 -eps_nev 4 -eps_ncv 20 -terse -dynamic_library_preload > $${test}.tmp 2>&1; \
	${TESTCODE}

runex2_ciss_1:
	-@${SETTEST}; check=ex2_ciss; \
	${MPIEXEC} -n 1 ./ex2 -n 30 -eps_type ciss -rg_type interval -rg_interval_endpoints 1.1,1.25 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex2_ciss_2:
	-@${SETTEST}; check=ex2_ciss; \
	${MPIEXEC} -n 2 ./ex2 -n 30 -eps_type ciss -rg_type ellipse -rg_ellipse_center 1.175 -rg_ellipse_radius 0.075 -eps_ciss_partitions 2 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex3_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex3 -n 72 -eps_nev 4 -eps_ncv 20 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex4_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex4 -file ${DATAPATH}/rdb200.petsc -eps_nev 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex4_ciss_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex4 -file $(DATAFILESPATH)/matrices/complex/qc324.petsc -eps_type ciss -rg_type ellipse -rg_ellipse_center -.012-.08i -rg_ellipse_radius .05 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex5_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex5 -eps_largest_real -eps_nev 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex6f90_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex6f90 -eps_max_it 1000 -eps_ncv 12 -eps_tol 1e-5 -eps_nev 4 -eps_largest_imaginary -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex7_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex7 -f1 ${DATAPATH}/bfw62a.petsc -f2 ${DATAPATH}/bfw62b.petsc -eps_nev 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex7_ciss_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex7 -f1 $(DATAFILESPATH)/matrices/complex/mhd1280a.petsc -f2 $(DATAFILESPATH)/matrices/complex/mhd1280b.petsc -eps_type ciss -eps_ciss_usest 0 -eps_ciss_quadrule chebyshev -rg_type ring -rg_ring_center 0 -rg_ring_radius .5 -rg_ring_width 0.2 -rg_ring_startangle .25 -rg_ring_endangle .5 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex9_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex9 -eps_nev 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex9_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex9 -eps_nev 8 -eps_max_it 300 -eps_target -28 -rg_type interval -rg_interval_endpoints -40,-20,-.1,.1 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex9_3:
	-@${SETTEST}; check=ex9_1; \
	${MPIEXEC} -n 1 ./ex9 -eps_nev 4 -eps_balance oneside -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex9_4:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex9 -eps_smallest_imaginary -eps_ncv 24 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex9_5:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex9 -eps_nev 4 -eps_target_real -eps_target -3 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex9_6:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex9 -eps_nev 2 -eps_target_imaginary -eps_target 3i -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex9_7:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex9 -n 40 -eps_nev 1 -eps_type arnoldi -eps_smallest_real -eps_refined -eps_max_it 200 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex9_8:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex9 -eps_nev 2 -eps_target -30 -eps_type jd -st_matmode shell -eps_jd_fix 0.0001 -eps_jd_const_correction_tol 0 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex10_1: runex10_1_sinvert runex10_1_sinvert_twoside runex10_1_shell runex10_1_shell_twoside
runex10_1_%:
	-@${SETTEST}; check=ex10_1; st=$*; \
	if [ "$$st" = sinvert_twoside ]; then st="sinvert -eps_balance twoside"; \
	elif [ "$$st" = shell_twoside ]; then st="shell -eps_balance twoside"; fi; \
	${MPIEXEC} -n 1 ./ex10 -eps_nev 5 -eps_non_hermitian -st_type $$st -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex11_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex11 -eps_nev 4 -eps_krylovschur_restart .2 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex11_2:
	-@${SETTEST}; check=ex11_1; \
	${MPIEXEC} -n 1 ./ex11 -eps_nev 4 -eps_ncv 20 -eps_target 0 -st_type sinvert -st_ksp_type cg -st_pc_type jacobi -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex12_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex12 -showinertia 0 -eps_error_relative > $${test}.tmp 2>&1; \
	${TESTCODE}

runex13_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex13 -eps_nev 4 -eps_ncv 22 -eps_tol 1e-5 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex13_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex13 -n 110 -nulldim 6 -eps_nev 4 -eps_ncv 18 -eps_tol 1e-5 -eps_purify 1 -st_type sinvert -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex18_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex18 -eps_nev 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex19_1: runex19_1_krylovschur runex19_1_lobpcg
runex19_1_%:
	-@${SETTEST}; check=ex19_1; eps=$*; \
	if [ "$$eps" = krylovschur ]; then EXTRA="-eps_ncv 64"; \
	else EXTRA="-eps_tol 1e-7"; fi; \
	${MPIEXEC} -n 1 ./ex19 -eps_nev 8 -eps_type $$eps $$EXTRA -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex24_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex24 -n 15 -eps_nev 1 -eps_ncv 12 -eps_max_it 1000 -eps_tol 1e-5 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex25_1:
	-@${SETTEST}; \
	if [ "${PETSC_PRECISION}" = "single" ]; then EXTRA="-eps_tol 1e-5"; fi; \
	${MPIEXEC} -n 1 ./ex25 -terse $$EXTRA > $${test}.tmp 2>&1; \
	${TESTCODE}

runex29_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex29 -m 350 > $${test}.tmp 2>&1; \
	${TESTCODE}

runex30_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex30 -n 100 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex31_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex31 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex34_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex34 -petscspace_order 1 -petscspace_poly_tensor > $${test}.tmp 2>&1; \
	${TESTCODE}

runex34_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex34 -petscspace_order 1 -petscspace_poly_tensor -eps_power_update > $${test}.tmp 2>&1; \
	${TESTCODE}

runex35_1:
	-@${SETTEST}; \
	if [ "${PETSC_PRECISION}" = "single" ]; then EXTRA="-eps_tol 1e-5"; fi; \
	${MPIEXEC} -n 1 ./ex35 -m 11 -eps_nev 4 -terse $$EXTRA > $${test}.tmp 2>&1; \
	${TESTCODE}

runex36_1:
	-@${SETTEST}; \
	if ${GREP} "\#define PETSC_HAVE_C99_COMPLEX" ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h > /dev/null; then \
	  if [ "${PETSC_PRECISION}" = "__float128" ]; then EXTRA="-eps_tol 1e-14"; fi; \
	  ${MPIEXEC} -n 1 ./ex36 -eps_nev 4 -mfn_ncv 16 -terse $$EXTRA | ${SED} -e "s/-2/+2/g" > $${test}.tmp 2>&1; \
	  ${TESTCODE}; \
	fi

