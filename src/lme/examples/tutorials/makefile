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
LOCDIR     = src/lme/examples/tutorials/
EXAMPLESC  = ex32.c
EXAMPLESF  =
MANSEC     = LME

TESTEXAMPLES_C_NOTSINGLE = ex32.PETSc runex32_1 ex32.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex32: ex32.o chkopts
	-${CLINKER} -o ex32 ex32.o ${SLEPC_LME_LIB}
	${RM} ex32.o

#------------------------------------------------------------------------------------

runex32_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex32 > $${test}.tmp 2>&1; \
	${TESTCODE}

