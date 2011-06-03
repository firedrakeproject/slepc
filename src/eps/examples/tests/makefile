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
LOCDIR     = src/eps/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c
EXAMPLESF  = 
MANSEC     = EPS
TESTS      = test1 test2 test3

TESTEXAMPLES_C       = test1.PETSc runtest1_1 test1.rm \
                       test2.PETSc runtest2_1 test2.rm \
                       test3.PETSc runtest3_1 test3.rm 

include ${SLEPC_DIR}/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_LIB}
	${RM} test2.o

test3: test3.o chkopts
	-${CLINKER} -o test3 test3.o ${SLEPC_LIB}
	${RM} test3.o

#------------------------------------------------------------------------------------
EPS = krylovschur arnoldi lanczos gd jd
runtest1_1:
	-@touch test1_1.tmp; \
	for eps in ${EPS}; do \
	   echo "eps type $$eps" >> test1_1.tmp; \
	   ${MPIEXEC} -np 1 ./test1 -eps_type $$eps >> test1_1.tmp 2>&1; \
	done; \
	if (${DIFF} output/test1_1.out test1_1.tmp) then true; \
	else echo "Possible problem with test1_1, diffs above"; fi; \
	${RM} -f test1_1.tmp

runtest2_1:
	-@touch test1_1.tmp; \
	for eps in ${EPS}; do \
	   echo "eps type $$eps" >> test1_1.tmp; \
	   ${MPIEXEC} -np 1 ./test2 -eps_type $$eps > test2_1.tmp 2>&1;    \
	done; \
	if (${DIFF} output/test2_1.out test2_1.tmp) then true; \
	else echo "Possible problem with test2_1, diffs above"; fi; \
	${RM} -f test2_1.tmp

runtest3_1:
	-@${MPIEXEC} -np 1 ./test3 > test3_1.tmp 2>&1;    \
	   if (${DIFF} output/test3_1.out test3_1.tmp) then true; \
	   else echo "Possible problem with test3_1, diffs above"; fi; \
	   ${RM} -f test3_1.tmp

