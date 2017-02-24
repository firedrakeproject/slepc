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
LOCDIR     = src/nep/examples/nlevp/
EXAMPLESC  = loaded_string.c gun.c
EXAMPLESF  =
MANSEC     = NEP
TESTS      = loaded_string gun

TESTEXAMPLES_C                     = loaded_string.PETSc runloaded_string_1 loaded_string.rm
TESTEXAMPLES_C_NOTSINGLE           = loaded_string.PETSc runloaded_string_2 runloaded_string_3 \
                                                         runloaded_string_4 loaded_string.rm
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
	${MPIEXEC} -n 1 ./loaded_string -nep_type interpol -rg_type interval -rg_interval_endpoints 5,700,-.1,.1 -nep_nev 7 -nep_target 5 -nep_interpol_degree 12 -nep_refine simple -nep_refine_scheme $$scheme -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runloaded_string_3: runloaded_string_3_explicit runloaded_string_3_mbe
runloaded_string_3_%:
	-@${SETTEST}; check=loaded_string_2; scheme=$*; \
	if [ "$$scheme" = mbe ]; then scheme="mbe -nep_refine_ksp_type preonly -nep_refine_pc_type cholesky"; fi; \
	${MPIEXEC} -n 2 ./loaded_string -nep_type interpol -rg_type interval -rg_interval_endpoints 5,700,-.1,.1 -nep_nev 7 -nep_target 5 -nep_interpol_degree 12 -nep_refine simple -nep_refine_partitions 2 -nep_refine_scheme $$scheme -nep_interpol_st_ksp_type bcgs -nep_interpol_st_pc_type bjacobi -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runloaded_string_4:
	-@${SETTEST}; check=loaded_string_2; \
	${MPIEXEC} -n 4 ./loaded_string -nep_type interpol -rg_type interval -rg_interval_endpoints 5,700,-.1,.1 -nep_nev 7 -nep_target 5 -nep_interpol_degree 10 -nep_refine simple -nep_refine_partitions 2 -nep_refine_scheme explicit -nep_interpol_st_ksp_type bcgs -nep_interpol_st_pc_type bjacobi -terse -info_exclude nep,pep,fn -log_exclude nep,pep,fn > $${test}.tmp 2>&1; \
	${TESTCODE}

rungun_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./gun -K $(DATAFILESPATH)/matrices/complex/gun_K.petsc -M $(DATAFILESPATH)/matrices/complex/gun_M.petsc -W1 $(DATAFILESPATH)/matrices/complex/gun_W1.petsc -W2 $(DATAFILESPATH)/matrices/complex/gun_W2.petsc -nep_type nleigs -rg_type interval -rg_interval_endpoints 12500,120500,-1,25000 -nep_target 65000 -nep_nev 4 -nep_tol 1e-12 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

