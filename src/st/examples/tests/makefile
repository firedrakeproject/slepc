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
LOCDIR     = src/st/examples/tests/
EXAMPLESC  = test1.c
EXAMPLESF  = 
MANSEC     = ST

TESTEXAMPLES_C       = test1.PETSc runtest1_1 test1.rm

include ${SLEPC_DIR}/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_LIB}
	${RM} test1.o

#------------------------------------------------------------------------------------

EPS = krylovschur arnoldi lanczos gd jd
runtest1_1:
	-@touch test1_1.tmp; \
	for eps in ${EPS}; do \
	   ${MPIEXEC} -np 1 ./test1 -eps_type $$eps -st_shift 1 -eps_smallest_real -st_matmode inplace -st_pc_type jacobi -eps_nev 4 -eps_terse >> test1_1.tmp 2>&1; \
	done; \
	if (${DIFF} output/test1_1.out test1_1.tmp) then true; \
	else echo "Possible problem with test1_1, diffs above"; fi; \
	${RM} -f test1_1.tmp

runtest1_2:
	-@touch test1_2.tmp; \
	for eps in ${EPS}; do \
	   ${MPIEXEC} -np 2 ./test1 -eps_type $$eps -st_shift 1 -eps_smallest_real -st_matmode inplace -st_pc_type jacobi -eps_nev 4 -eps_terse >> test1_2.tmp 2>&1; \
	done; \
	if (${DIFF} output/test1_2.out test1_2.tmp) then true; \
	else echo "Possible problem with test1_2, diffs above"; fi; \
	${RM} -f test1_2.tmp

