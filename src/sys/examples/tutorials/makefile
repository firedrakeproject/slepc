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
LOCDIR     = src/sys/examples/tutorials/
EXAMPLESC  = ex33.c
EXAMPLESF  =
MANSEC     = sys

TESTEXAMPLES_C = ex33.PETSc runex33_1 ex33.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex33: ex33.o chkopts
	-${CLINKER} -o ex33 ex33.o ${SLEPC_SYS_LIB}
	${RM} ex33.o

#------------------------------------------------------------------------------------

runex33_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./ex33 > $${test}.tmp 2>&1; \
	${TESTCODE}

