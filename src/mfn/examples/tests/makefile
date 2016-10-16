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
EXAMPLESC  = test1.c test2.c test3.c
EXAMPLESF  = test3f.F
MANSEC     = MFN
TESTS      = test1 test2 test3 test3f

TESTEXAMPLES_C           = test2.PETSc runtest2_1 test2.rm \
                           test3.PETSc runtest3_1 test3.rm
TESTEXAMPLES_C_DATAFILE  = test1.PETSc runtest1_1 test1.rm
TESTEXAMPLES_FORTRAN     = test3f.PETSc runtest3f_1 test3f.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_MFN_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_MFN_LIB}
	${RM} test2.o

test3: test3.o chkopts
	-${CLINKER} -o test3 test3.o ${SLEPC_MFN_LIB}
	${RM} test3.o

test3f: test3f.o chkopts
	-${FLINKER} -o test3f test3f.o ${SLEPC_MFN_LIB}
	${RM} test3f.o

#------------------------------------------------------------------------------------
DATAPATH = ${SLEPC_DIR}/share/slepc/datafiles/matrices

runtest1_1: runtest1_1_krylov runtest1_1_expokit
runtest1_1_%:
	-@${SETTEST}; check=test1; mfn=$*; \
	${MPIEXEC} -n 1 ./test1 -file ${DATAPATH}/bfw62b.petsc -mfn_type $$mfn > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1: runtest2_1_krylov runtest2_1_expokit
runtest2_1_%:
	-@${SETTEST}; check=test2; mfn=$*; \
	${MPIEXEC} -n 1 ./test2 -mfn_type $$mfn > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test3 -myprefix_mfn_monitor_cancel -myprefix_mfn_converged_reason -myprefix_mfn_view >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test3f -info_exclude mfn -log_exclude mfn >> $${test}.tmp 2>&1; \
	${TESTCODE}

