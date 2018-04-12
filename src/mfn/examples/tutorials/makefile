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
LOCDIR     = src/mfn/examples/tutorials/
EXAMPLESC  = ex23.c ex26.c ex37.c
EXAMPLESF  = ex23f90.F90
MANSEC     = MFN

TESTEXAMPLES_C   = ex23.PETSc runex23_1 ex23.rm \
                   ex26.PETSc runex26_1 ex26.rm \
                   ex37.PETSc runex37_1 ex37.rm
TESTEXAMPLES_F90 = ex23f90.PETSc runex23f90_1 ex23f90.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex23: ex23.o chkopts
	-${CLINKER} -o ex23 ex23.o ${SLEPC_MFN_LIB}
	${RM} ex23.o

ex23f90: ex23f90.o chkopts
	-${FLINKER} -o ex23f90 ex23f90.o ${SLEPC_MFN_LIB}
	${RM} ex23f90.o

ex26: ex26.o chkopts
	-${CLINKER} -o ex26 ex26.o ${SLEPC_MFN_LIB}
	${RM} ex26.o

ex37: ex37.o chkopts
	-${CLINKER} -o ex37 ex37.o ${SLEPC_MFN_LIB}
	${RM} ex37.o
#------------------------------------------------------------------------------------

runex23_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex23 -mfn_ncv 6 > $${test}.tmp 2>&1; \
	${TESTCODE}

runex23f90_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex23f90 -mfn_ncv 6 > $${test}.tmp 2>&1; \
	${TESTCODE}

runex26_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex26 -mfn_tol 1e-5 > $${test}.tmp 2>&1; \
	${TESTCODE}

runex37_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex37 -mfn_tol 1e-6 > $${test}.tmp 2>&1; \
	${TESTCODE}
