#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain
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
                       test2.PETSc runtest2_1 runtest2_2 test2.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_PEP_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_PEP_LIB}
	${RM} test2.o

#------------------------------------------------------------------------------------
PEP = toar qarnoldi
EPS = krylovschur gd

TESTCODE = \
	[ x${SAVE_OUTPUT} = xyes ] && cp $${test}.tmp output/$${test}.out; \
	${DIFF} output/$${test}.out $${test}.tmp || \
	echo "Possible problem with $${test}, diffs above"; \
	${RM} -f $${test}.tmp

runtest1_1:
	-@test=test1_1; \
	for pep in ${PEP}; do \
	   echo "pep type $$pep"; \
	   ${MPIEXEC} -np 1 ./test1 -type $$pep -m 11 2>&1; \
	done > $${test}.tmp; \
	for eps in ${EPS}; do \
	    echo "pep type linear, eps type $$eps"; \
	    ${MPIEXEC} -np 1 ./test1 -type linear -epstype $$eps -m 11 2>&1; \
	done >> $${test}.tmp; \
	${TESTCODE}

runtest2_1:
	-@test=test2_1; \
	for pep in ${PEP}; do \
	   echo "pep type $$pep"; \
	   if [ $$pep = qarnoldi ]; then EXTRA="-st_transform"; else EXTRA=""; fi; \
	   ${MPIEXEC} -np 1 ./test2 -pep_type $$pep -pep_nev 4 $$EXTRA 2>&1; \
	done > $${test}.tmp; \
	for eps in ${EPS}; do \
	   echo "pep type linear, eps type $$eps"; \
	   ${MPIEXEC} -np 1 ./test2 -pep_type linear -pep_eps_type $$eps -pep_nev 4 -pep_linear_explicitmatrix 2>&1; \
	done >> $${test}.tmp; \
	${TESTCODE}

runtest2_2:
	-@test=test2_2; \
	echo "toar with shift-and-invert" > $${test}.tmp; \
	${MPIEXEC} -np 1 ./test2 -pep_type toar -st_type sinvert -pep_target -0.4 -pep_nev 4 2>&1 >> $${test}.tmp; \
	echo "toar with shift-and-invert (with scale=both)" >> $${test}.tmp; \
	${MPIEXEC} -np 1 ./test2 -pep_type toar -st_type sinvert -pep_target -0.4 -pep_nev 4 -pep_scale both 2>&1 >> $${test}.tmp; \
	echo "toar with shift-and-invert (transform)" >> $${test}.tmp; \
	${MPIEXEC} -np 1 ./test2 -pep_type toar -st_type sinvert -pep_target -0.4 -pep_nev 4 -st_transform 2>&1 >> $${test}.tmp; \
	echo "qarnoldi with shift-and-invert (transform)" >> $${test}.tmp; \
	${MPIEXEC} -np 1 ./test2 -pep_type qarnoldi -st_type sinvert -pep_target -0.4 -pep_nev 4 -st_transform 2>&1 >> $${test}.tmp; \
	echo "linear with shift-and-invert" >> $${test}.tmp; \
	${MPIEXEC} -np 1 ./test2 -pep_type linear -pep_target -0.4 -pep_nev 4 -st_type sinvert 2>&1 >> $${test}.tmp; \
	${TESTCODE}

