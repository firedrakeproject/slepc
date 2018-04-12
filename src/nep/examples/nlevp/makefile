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
LOCDIR     = src/nep/examples/nlevp/
EXAMPLESC  = loaded_string.c gun.c
EXAMPLESF  =
MANSEC     = NEP
TESTS      = loaded_string gun

TESTEXAMPLES_C                     = loaded_string.PETSc runloaded_string_1 loaded_string.rm
TESTEXAMPLES_C_NOTSINGLE           = loaded_string.PETSc runloaded_string_2 runloaded_string_3 \
                                                         runloaded_string_4 runloaded_string_5 \
                                                         runloaded_string_6 loaded_string.rm
TESTEXAMPLES_C_DOUBLE              = loaded_string.PETSc runloaded_string_7 loaded_string.rm
TESTEXAMPLES_DATAFILESPATH_COMPLEX = gun.PETSc rungun_1 gun.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

loaded_string: loaded_string.o chkopts
	-${CLINKER} -o loaded_string loaded_string.o ${SLEPC_NEP_LIB}
	${RM} loaded_string.o

gun: gun.o chkopts
	-${CLINKER} -o gun gun.o ${SLEPC_NEP_LIB}
	${RM} gun.o

#------------------------------------------------------------------------------------
runloaded_string_1:
	-@${SETTEST}; \
	if [ "${PETSC_PRECISION}" = "single" ]; then EXTRA="-nep_tol 1e-5"; fi; \
	${MPIEXEC} -n 1 ./loaded_string -nep_type rii -terse $$EXTRA > $${test}.tmp 2>&1; \
	${TESTCODE}

runloaded_string_2: runloaded_string_2_schur runloaded_string_2_explicit runloaded_string_2_mbe
runloaded_string_2_%:
	-@${SETTEST}; check=loaded_string_2; scheme=$*; \
	if [ "$$scheme" = mbe ]; then scheme="mbe -nep_refine_ksp_type preonly -nep_refine_pc_type lu"; fi; \
	${MPIEXEC} -n 1 ./loaded_string -nep_type interpol -rg_type interval -rg_interval_endpoints 5,700,-.1,.1 -nep_nev 7 -nep_target 5 -nep_interpol_interpolation_degree 12 -nep_refine simple -nep_refine_scheme $$scheme -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runloaded_string_3: runloaded_string_3_explicit runloaded_string_3_mbe
runloaded_string_3_%:
	-@${SETTEST}; check=loaded_string_2; scheme=$*; \
	if [ "$$scheme" = mbe ]; then scheme="mbe -nep_refine_ksp_type preonly -nep_refine_pc_type cholesky"; fi; \
	${MPIEXEC} -n 2 ./loaded_string -nep_type interpol -rg_type interval -rg_interval_endpoints 5,700,-.1,.1 -nep_nev 7 -nep_target 5 -nep_interpol_interpolation_degree 12 -nep_refine simple -nep_refine_partitions 2 -nep_refine_scheme $$scheme -nep_interpol_st_ksp_type bcgs -nep_interpol_st_pc_type bjacobi -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runloaded_string_4:
	-@${SETTEST}; check=loaded_string_2; \
	${MPIEXEC} -n 4 ./loaded_string -nep_type interpol -rg_type interval -rg_interval_endpoints 5,700,-.1,.1 -nep_nev 7 -nep_target 5 -nep_interpol_interpolation_degree 10 -nep_refine simple -nep_refine_partitions 2 -nep_refine_scheme explicit -nep_interpol_st_ksp_type bcgs -nep_interpol_st_pc_type bjacobi -terse -info_exclude nep,pep,fn -log_exclude nep,pep,fn > $${test}.tmp 2>&1; \
	${TESTCODE}

runloaded_string_5:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./loaded_string -nep_type nleigs -rg_type interval -rg_interval_endpoints 4,700,-.1,.1 -nep_nev 8 -nep_target 5 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runloaded_string_6:
	-@${SETTEST}; \
        if [ "${PETSC_SCALAR}" = "complex" ]; then region=",-.1,.1"; fi; \
	${MPIEXEC} -n 1 ./loaded_string -nep_type nleigs -rg_type interval -rg_interval_endpoints 100,700$$region -nep_nev 5 -nep_tol 1e-9 -nep_target 140 -nep_nleigs_interpolation_degree 15 -nep_general -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runloaded_string_7:
	-@${SETTEST}; \
        if [ "${PETSC_SCALAR}" = "complex" ]; then region=",-.1,.1"; fi; \
	${MPIEXEC} -n 1 ./loaded_string -nep_type interpol -rg_type interval -rg_interval_endpoints 5,700$$region -nep_nev 5 -nep_target 100 -nep_interpol_interpolation_degree 20 -nep_ncv 20 -n 20 -nep_refine simple -nep_refine_its 1 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

rungun_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./gun -K $(DATAFILESPATH)/matrices/complex/gun_K.petsc -M $(DATAFILESPATH)/matrices/complex/gun_M.petsc -W1 $(DATAFILESPATH)/matrices/complex/gun_W1.petsc -W2 $(DATAFILESPATH)/matrices/complex/gun_W2.petsc -nep_type nleigs -rg_type polygon -rg_polygon_vertices 12500-1i,120500-1i,120500+30000i,70000+30000i -nep_target 65000 -nep_nev 24 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

