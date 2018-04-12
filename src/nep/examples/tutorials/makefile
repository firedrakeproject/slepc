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
LOCDIR     = src/nep/examples/tutorials/
EXAMPLESC  = ex20.c ex21.c ex22.c ex27.c
EXAMPLESF  = ex20f90.F90 ex22f90.F90 ex27f90.F90
MANSEC     = NEP

TESTEXAMPLES_C             = ex27.PETSc runex27_1 runex27_2 runex27_4 ex27.rm
TESTEXAMPLES_C_NOTSINGLE   = ex20.PETSc runex20_1 ex20.rm \
                             ex21.PETSc runex21_1 ex21.rm \
                             ex22.PETSc runex22_1 runex22_2 ex22.rm \
                             ex27.PETSc runex27_3 runex27_9 ex27.rm
TESTEXAMPLES_C_COMPLEX     = ex22.PETSc runex22_1_ciss ex22.rm \
                             ex27.PETSc runex27_7 runex27_8 ex27.rm
TESTEXAMPLES_VECCUDA       = ex27.PETSc runex27_5_cuda runex27_6_cuda ex27.rm
TESTEXAMPLES_F90_NOTSINGLE = ex20f90.PETSc runex20f90_1 ex20f90.rm \
                             ex22f90.PETSc runex22f90_1 ex22f90.rm \
                             ex27f90.PETSc runex27f90_1 ex27f90.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex20: ex20.o chkopts
	-${CLINKER} -o ex20 ex20.o ${SLEPC_NEP_LIB}
	${RM} ex20.o

ex20f90: ex20f90.o chkopts
	-${FLINKER} -o ex20f90 ex20f90.o ${SLEPC_NEP_LIB}
	${RM} ex20f90.o usermodule.mod

ex21: ex21.o chkopts
	-${CLINKER} -o ex21 ex21.o ${SLEPC_NEP_LIB}
	${RM} ex21.o

ex22: ex22.o chkopts
	-${CLINKER} -o ex22 ex22.o ${SLEPC_NEP_LIB}
	${RM} ex22.o

ex22f90: ex22f90.o chkopts
	-${FLINKER} -o ex22f90 ex22f90.o ${SLEPC_NEP_LIB}
	${RM} ex22f90.o

ex27: ex27.o chkopts
	-${CLINKER} -o ex27 ex27.o ${SLEPC_NEP_LIB}
	${RM} ex27.o

ex27f90: ex27f90.o chkopts
	-${FLINKER} -o ex27f90 ex27f90.o ${SLEPC_NEP_LIB}
	${RM} ex27f90.o

#------------------------------------------------------------------------------------

runex20_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex20 | ${SED} -e "s/[0-9]\.[0-9]*e-[0-9]*/removed/g" -e "s/\( Number of NEP iterations =\) [0-9]*/\1/" > $${test}.tmp 2>&1; \
	${TESTCODE}

runex20f90_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex20f90 | ${SED} -e "s/[0-9]\.[0-9]*E-[0-9]*/removed/g" -e "s/\( Number of NEP iterations =\) [ 0-9]*/\1/" > $${test}.tmp 2>&1; \
	${TESTCODE}

runex21_1: runex21_1_rii runex21_1_slp
runex21_1_%:
	-@${SETTEST}; check=ex21_1; nep=$*; \
	if [ "$$nep" = slp ]; then nep="slp -nep_slp_st_matmode shell -nep_slp_st_ksp_type bcgs -nep_target 10"; fi; \
	${MPIEXEC} -n 1 ./ex21 -nep_type $${nep} -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex22_1: runex22_1_rii runex22_1_slp runex22_1_narnoldi
runex22_1_%:
	-@${SETTEST}; check=ex22_1; nep=$*; \
	${MPIEXEC} -n 1 ./ex22 -nep_type $${nep} -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex22_1_ciss:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex22 -nep_type ciss -rg_type ellipse -rg_ellipse_center 10 -rg_ellipse_radius 9.5 -nep_ncv 24 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex22_2: runex22_2_none runex22_2_norm runex22_2_residual
runex22_2_%:
	-@${SETTEST}; check=ex22_2; extract=$*; \
	${MPIEXEC} -n 1 ./ex22 -nep_type interpol -nep_interpol_pep_extract $$extract -rg_type interval -rg_interval_endpoints 5,20,-.1,.1 -nep_nev 3 -nep_target 5 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex22f90_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex22f90 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex27_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex27 -nep_nev 3 -nep_nleigs_interpolation_degree 90 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex27_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex27 -split 0 -nep_nev 3 -nep_nleigs_interpolation_degree 90 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex27_3:
	-@${SETTEST}; check=ex27_1; \
	${MPIEXEC} -n 1 ./ex27 -nep_nev 3 -nep_tol 1e-8 -nep_nleigs_rk_shifts 1.06,1.1,1.12,1.15 -nep_conv_norm -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex27_4:
	-@${SETTEST}; check=ex27_2; \
	${MPIEXEC} -n 1 ./ex27 -split 0 -nep_nev 3 -nep_nleigs_rk_shifts 1.06,1.1,1.12,1.15 -nep_nleigs_interpolation_degree 90 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex27_5_cuda:
	-@${SETTEST}; check=ex27_1; \
	${MPIEXEC} -n 1 ./ex27 -nep_nev 3 -mat_type aijcusparse -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex27_6_cuda:
	-@${SETTEST}; check=ex27_2; \
	${MPIEXEC} -n 1 ./ex27 -split 0 -nep_nev 3 -mat_type aijcusparse -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex27_7:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex27 -split 0 -nep_type ciss -rg_type ellipse -rg_ellipse_center 8 -rg_ellipse_radius .7 -rg_ellipse_vscale 0.1 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex27_8:
	-@${SETTEST}; check=ex27_7; \
	${MPIEXEC} -n 1 ./ex27 -nep_type ciss -rg_type ellipse -rg_ellipse_center 8 -rg_ellipse_radius .7 -rg_ellipse_vscale 0.1 -terse | ${SED} -e "s/ (in split form)//" > $${test}.tmp 2>&1; \
	${TESTCODE}

runex27_9:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex27 -nep_nev 4 -n 20 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex27f90_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex27f90 -nep_nev 3 -nep_nleigs_interpolation_degree 90 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex27f90_2:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex27f90 -split 0 -nep_nev 3 -nep_nleigs_interpolation_degree 90 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

