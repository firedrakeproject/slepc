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
LOCDIR     = src/mfn/examples/tests/
EXAMPLESC  = test1.c test2.c
EXAMPLESF  =
MANSEC     = MFN
TESTS      = test1 test2

TESTEXAMPLES_C           = test2.PETSc runtest2_1 test2.rm
TESTEXAMPLES_C_DATAFILE  = test1.PETSc runtest1_1 test1.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_MFN_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_MFN_LIB}
	${RM} test2.o

#------------------------------------------------------------------------------------
DATAPATH = ${SLEPC_DIR}/share/slepc/datafiles/matrices

runtest1_1: runtest1_1_krylov runtest1_1_expokit
runtest1_1_%:
	-@test=$(subst run,,$@); check=test1; mfn=$*; \
	${MPIEXEC} -n 1 ./test1 -file ${DATAPATH}/bfw62b.petsc -mfn_type $$mfn > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1: runtest2_1_krylov runtest2_1_expokit
runtest2_1_%:
	-@test=$(subst run,,$@); check=test2; mfn=$*; \
	${MPIEXEC} -n 1 ./test2 -mfn_type $$mfn > $${test}.tmp 2>&1; \
	${TESTCODE}

