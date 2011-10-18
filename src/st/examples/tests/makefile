#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain
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
LOCDIR     = src/st/examples/tests/
EXAMPLESC  = test1.c
EXAMPLESF  = 
MANSEC     = ST
TESTS      = test1

TESTEXAMPLES_C       = test1.PETSc runtest1_1 runtest1_2 test1.rm

include ${SLEPC_DIR}/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_LIB}
	${RM} test1.o

#------------------------------------------------------------------------------------

EPS = krylovschur arnoldi gd jd

TESTCODE = \
	[ x${SAVE_OUTPUT} = xyes ] && cp $${test}.tmp output/$${test}.out; \
	${DIFF} output/$${test}.out $${test}.tmp || \
	echo "Possible problem with $${test}, diffs above"; \
	${RM} -f $${test}.tmp

runtest1_1:
	-@test=test1_1; \
	for eps in ${EPS}; do \
	   echo "eps type $$eps"; \
	   ${MPIEXEC} -np 1 ./test1 -eps_type $$eps -st_shift 1 -eps_smallest_real -st_matmode inplace -st_pc_type jacobi -eps_nev 4 -eps_terse 2>&1; \
	done > $${test}.tmp; \
	${TESTCODE}

runtest1_2:
	-@test=test1_2; \
	for eps in ${EPS}; do \
	   echo "eps type $$eps"; \
	   ${MPIEXEC} -np 2 ./test1 -eps_type $$eps -st_shift 1 -eps_smallest_real -st_matmode inplace -st_pc_type jacobi -eps_nev 4 -eps_terse 2>&1; \
	done > $${test}.tmp; \
	${TESTCODE}
