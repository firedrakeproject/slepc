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
LOCDIR     = src/svd/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c
EXAMPLESF  =
MANSEC     = SVD
TESTS      = test1 test2 test3

TESTEXAMPLES_C       = test1.PETSc runtest1_1 test1.rm \
                       test3.PETSc runtest3_1 test3.rm
TESTEXAMPLES_C_NOF128= test2.PETSc runtest2_1 test2.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_SVD_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_SVD_LIB}
	${RM} test2.o

test3: test3.o chkopts
	-${CLINKER} -o test3 test3.o ${SLEPC_SVD_LIB}
	${RM} test3.o

#------------------------------------------------------------------------------------
SVD = lanczos lapack trlanczos
SVDEPS = cross cyclic
EPS = krylovschur arnoldi lanczos gd jd

runtest1_1:
	-@touch test1_1.tmp; \
	for svd in ${SVDEPS} ${SVD}; do \
	   ${MPIEXEC} -n 1 ./test1 -type $$svd >> test1_1.tmp 2>&1; \
	done; \
	for svd in ${SVDEPS}; do \
	   for eps in ${EPS}; do \
	      ${MPIEXEC} -n 1 ./test1 -type $$svd -epstype $$eps >> test1_1.tmp 2>&1; \
           done; \
	done; \
	if (${DIFF} output/test1_1.out test1_1.tmp) then true; \
	else echo "Possible problem with test1_1, diffs above"; fi; \
	${RM} -f test1_1.tmp

runtest2_1:
	-@touch test2_1.tmp; \
	for svd in ${SVDEPS} ${SVD}; do \
	   echo "svd type $$svd" >> test2_1.tmp; \
	   ${MPIEXEC} -n 1 ./test2 -svd_type $$svd -svd_nsv 7 >> test2_1.tmp 2>&1;    \
	done; \
	if (${GREP} USE_COMPLEX ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h > /dev/null 2>&1) then \
	  [ x${SAVE_OUTPUT} = xyes ] && cp test2_1.tmp output/test2_1_complex.out; \
	  if (${DIFF} output/test2_1_complex.out test2_1.tmp) then true; \
	  else echo "Possible problem with test2_1, diffs above"; fi; \
	else \
	  [ x${SAVE_OUTPUT} = xyes ] && cp test2_1.tmp output/test2_1.out; \
	  if (${DIFF} output/test2_1.out test2_1.tmp) then true; \
	  else echo "Possible problem with test2_1, diffs above"; fi; \
	fi; \
	${RM} -f test2_1.tmp;

runtest3_1:
	-@touch test3_1.tmp; \
	for svd in ${SVDEPS} ${SVD}; do \
	   echo "svd type $$svd" >> test3_1.tmp; \
	   ${MPIEXEC} -n 1 ./test3 -svd_type $$svd -svd_nsv 4 >> test3_1.tmp 2>&1; \
	done; \
	echo "svd type lanczos one-sided" >> test3_1.tmp; \
	${MPIEXEC} -n 1 ./test3 -svd_type $$svd -svd_nsv 4 -svd_type lanczos -svd_lanczos_oneside >> test3_1.tmp 2>&1; \
	echo "svd type trlanczos one-sided" >> test3_1.tmp; \
	${MPIEXEC} -n 1 ./test3 -svd_type $$svd -svd_nsv 4 -svd_type trlanczos -svd_trlanczos_oneside >> test3_1.tmp 2>&1; \
	[ x${SAVE_OUTPUT} = xyes ] && cp test3_1.tmp output/test3_1.out; \
	if (${DIFF} output/test3_1.out test3_1.tmp) then true; \
	else echo "Possible problem with test3_1, diffs above"; fi; \
	${RM} -f test3_1.tmp

