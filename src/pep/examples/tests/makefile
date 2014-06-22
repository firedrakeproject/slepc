#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain
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
LOCDIR     = src/pep/examples/tests/
EXAMPLESC  = test1.c test2.c
EXAMPLESF  =
MANSEC     = PEP
TESTS      = test1 test2

TESTEXAMPLES_C       = test1.PETSc runtest1_1 test1.rm \
                       test2.PETSc runtest2_1 test2.rm

include ${SLEPC_DIR}/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_PEP_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_PEP_LIB}
	${RM} test2.o

#------------------------------------------------------------------------------------
PEP = qarnoldi
PEPEPS = linear
EPS = krylovschur

TESTCODE = \
	[ x${SAVE_OUTPUT} = xyes ] && cp $${test}.tmp output/$${test}.out; \
	${DIFF} output/$${test}.out $${test}.tmp || \
	echo "Possible problem with $${test}, diffs above"; \
	${RM} -f $${test}.tmp

runtest1_1:
	-@test=test1_1; \
	for pep in ${PEPEPS} ${PEP}; do \
	   echo "pep type $$pep"; \
	   ${MPIEXEC} -np 1 ./test1 -type $$pep -m 11 -pep_terse 2>&1; \
	done > $${test}.tmp; \
	for pep in ${PEPEPS}; do \
	   for eps in ${EPS}; do \
	      echo "pep type $$pep, eps type $$eps"; \
	      ${MPIEXEC} -np 1 ./test1 -type $$pep -epstype $$eps -m 11 -pep_terse 2>&1; \
           done; \
	done >> $${test}.tmp; \
	${TESTCODE}

runtest2_1:
	-@test=test2_1; \
	for pep in ${PEP}; do \
	   echo "pep type $$pep"; \
	   if [ $$pep = qarnoldi ]; then EXTRA="-pep_ncv 30"; else EXTRA=""; fi; \
	   ${MPIEXEC} -np 1 ./test2 -pep_type $$pep -pep_nev 4 -pep_terse $$EXTRA 2>&1; \
	done > $${test}.tmp; \
	for pep in ${PEPEPS}; do \
	   for eps in ${EPS}; do \
	      echo "pep type $$pep, eps type $$eps"; \
	      if [ $$eps = arnoldi -o $$eps = krylovschur ]; then EXTRA="-pep_ncv 30"; else EXTRA=""; fi; \
	      ${MPIEXEC} -np 1 ./test2 -pep_type $$pep -pep_eps_type $$eps -pep_nev 4 -pep_terse $$EXTRA 2>&1; \
           done; \
	done >> $${test}.tmp; \
	${TESTCODE}

