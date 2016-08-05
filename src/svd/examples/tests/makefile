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

TESTEXAMPLES_C           = test1.PETSc runtest1_1 test1.rm
TESTEXAMPLES_C_DOUBLE    = test2.PETSc runtest2_1 test2.rm
TESTEXAMPLES_C_NOTSINGLE = test3.PETSc runtest3_1 test3.rm

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

runtest1_1: runtest1_1_lanczos runtest1_1_trlanczos runtest1_1_cross runtest1_1_cross_gd runtest1_1_cyclic runtest1_1_cyclic_gd runtest1_1_lapack 
runtest1_1_%:
	-@test=$(subst run,,$@); check=test1_1; svd=$*; \
	if [ "$$svd" = cross_gd ]; then svd="cross -epstype gd"; fi; \
	if [ "$$svd" = cyclic_gd ]; then svd="cyclic -epstype gd"; fi; \
	${MPIEXEC} -n 1 ./test1 -type $$svd >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1: runtest2_1_lanczos runtest2_1_trlanczos runtest2_1_cross runtest2_1_cyclic runtest2_1_lapack 
runtest2_1_%:
	-@test=$(subst run,,$@); check=test2_1; svd=$*; \
	if [ "${PETSC_SCALAR}" = "complex" ]; then check=$${check}_complex; fi; \
	${MPIEXEC} -n 1 ./test2 -svd_type $$svd -svd_nsv 7 > $${test}.tmp 2>&1;    \
	${TESTCODE}

runtest3_1: runtest3_1_lanczos runtest3_1_lanczos_one runtest3_1_trlanczos runtest3_1_trlanczos_one runtest3_1_cross runtest3_1_cyclic runtest3_1_lapack 
runtest3_1_%:
	-@test=$(subst run,,$@); check=test3_1; svd=$*; \
	if [ "$$svd" = lanczos_one ]; then svd="lanczos -svd_lanczos_oneside"; fi; \
	if [ "$$svd" = trlanczos_one ]; then svd="trlanczos -svd_trlanczos_oneside"; fi; \
	${MPIEXEC} -n 1 ./test3 -svd_type $$svd -svd_nsv 4 > $${test}.tmp 2>&1; \
	${TESTCODE}

