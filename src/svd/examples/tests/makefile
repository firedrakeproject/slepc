#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain
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
LOCDIR     = src/svd/examples/tests/
EXAMPLESC  = 
EXAMPLESF  = 
MANSEC     = SVD

TESTEXAMPLES_C       = test1.PETSc runtest1_1 test1.rm

include ${SLEPC_DIR}/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_LIB}
	${RM} test1.o

#------------------------------------------------------------------------------------
SVD = lanczos lapack trlanczos
SVDEPS = cross cyclic
EPS = krylovschur arnoldi lanczos gd jd

runtest1_1:
	-@touch test1_1.tmp; \
	for svd in ${SVDEPS} ${SVD}; do \
	   echo "svd type $$svd" >> test1_1.tmp; \
	   ${MPIEXEC} -np 1 ./test1 -type $$svd >> test1_1.tmp 2>&1; \
	done; \
	for svd in ${SVDEPS}; do \
	   for eps in ${EPS}; do \
	      echo "svd type $$svd eps type $$eps" >> test1_1.tmp; \
	      ${MPIEXEC} -np 1 ./test1 -type $$svd -epstype $$eps >> test1_1.tmp 2>&1; \
           done; \
	done; \
	if (${DIFF} output/test1_1.out test1_1.tmp) then true; \
	else echo "Possible problem with test1_1, diffs above"; fi; \
	${RM} -f test1_1.tmp

