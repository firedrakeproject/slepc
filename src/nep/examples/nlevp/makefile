#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain
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

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

loaded_string: loaded_string.o chkopts
	-${CLINKER} -o loaded_string loaded_string.o ${SLEPC_NEP_LIB}
	${RM} loaded_string.o

gun: gun.o chkopts
	-${CLINKER} -o gun gun.o ${SLEPC_NEP_LIB}
	${RM} gun.o
