#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     SLEPc - Scalable Library for Eigenvalue Problem Computations
#     Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain
#
#     This file is part of SLEPc. See the README file for conditions of use
#     and additional information.
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

include ${PETSC_DIR}/bmake/common/base

checklink: checklink.o chkopts
	${CLINKER} -o checklink checklink.o ${TESTFLAGS} ${PETSC_KSP_LIB}
	@${RM} -f checklink checklink.o
