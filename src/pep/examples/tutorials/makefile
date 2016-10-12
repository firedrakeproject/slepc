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
LOCDIR     = src/pep/examples/tutorials/
EXAMPLESC  = ex16.c ex17.c ex28.c
EXAMPLESF  = ex16f90.F90
MANSEC     = PEP

TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE = ex28.PETSc runex28_1 ex28.rm \
                                     ex16.PETSc runex16_1_stoar runex16_1_linear_symm ex16.rm
TESTEXAMPLES_C_NOCOMPLEX           = ex16.PETSc runex16_1 ex16.rm
TESTEXAMPLES_C_DATAFILE            = ex17.PETSc runex17_1 ex17.rm
TESTEXAMPLES_F90_NOCOMPLEX         = ex16f90.PETSc runex16f90_1 ex16f90.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex16: ex16.o chkopts
	-${CLINKER} -o ex16 ex16.o ${SLEPC_PEP_LIB}
	${RM} ex16.o

ex16f90: ex16f90.o chkopts
	-${FLINKER} -o ex16f90 ex16f90.o ${SLEPC_PEP_LIB}
	${RM} ex16f90.o

ex17: ex17.o chkopts
	-${CLINKER} -o ex17 ex17.o ${SLEPC_PEP_LIB}
	${RM} ex17.o

ex28: ex28.o chkopts
	-${CLINKER} -o ex28 ex28.o ${SLEPC_PEP_LIB}
	${RM} ex28.o

#------------------------------------------------------------------------------------
DATAPATH = ${SLEPC_DIR}/share/slepc/datafiles/matrices

runex16_1: runex16_1_toar runex16_1_qarnoldi runex16_1_linear
runex16_1_%:
	-@${SETTEST}; check=ex16_1; pep=$*; \
	if [ "$$pep" = linear ]; then pep="linear -pep_linear_explicitmatrix"; \
	elif [ "$$pep" = linear_symm ]; then pep="linear -pep_linear_explicitmatrix -pep_linear_eps_gen_indefinite"; fi; \
	${MPIEXEC} -n 1 ./ex16 -pep_type $$pep -pep_nev 4 -pep_ncv 21 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex16f90_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex16f90 -pep_nev 4 -pep_ncv 20 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex17_1: runex17_1_toar runex17_1_qarnoldi runex17_1_linear
runex17_1_%:
	-@${SETTEST}; check=ex17_1; pep=$*; \
	${MPIEXEC} -n 1 ./ex17 -A ${DATAPATH}/speaker107k.petsc,${DATAPATH}/speaker107c.petsc,${DATAPATH}/speaker107m.petsc -pep_type $$pep -pep_nev 4 -pep_ncv 20 -pep_scale scalar -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runex28_1: runex28_1_toar runex28_1_qarnoldi runex28_1_linear
runex28_1_%:
	-@${SETTEST}; check=ex28_1; pep=$*; \
	${MPIEXEC} -n 1 ./ex28 -pep_nev 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

