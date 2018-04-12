#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
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

TESTEXAMPLES_C                 = test2.PETSc runtest2_1 test2.rm \
                                 test3.PETSc runtest3_1 test3.rm
TESTEXAMPLES_C_DATAFILE        = test1.PETSc runtest1_1 test1.rm
TESTEXAMPLES_FORTRAN           = test3f.PETSc runtest3f_1 test3f.rm
TESTEXAMPLES_VECCUDA_NOCOMPLEX = test1.PETSc runtest1_2_cuda test1.rm \
                                 test2.PETSc runtest2_2_cuda test2.rm

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

runtest1_2_cuda: runtest1_2_cuda_krylov runtest1_2_cuda_expokit
runtest1_2_cuda_%:
	-@${SETTEST}; check=test1; mfn=$*; \
	${MPIEXEC} -n 1 ./test1 -file ${DATAPATH}/bfw62b.petsc -mfn_type $$mfn -mat_type aijcusparse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1: runtest2_1_krylov runtest2_1_expokit
runtest2_1_%:
	-@${SETTEST}; check=test2; mfn=$*; \
	${MPIEXEC} -n 1 ./test2 -mfn_type $$mfn > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_2_cuda: runtest2_2_cuda_krylov runtest2_2_cuda_expokit
runtest2_2_cuda_%:
	-@${SETTEST}; check=test2; mfn=$*; \
	${MPIEXEC} -n 1 ./test2 -mfn_type $$mfn -mat_type aijcusparse > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test3 -myprefix_mfn_monitor_cancel -myprefix_mfn_converged_reason -myprefix_mfn_view >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test3f -info_exclude mfn -log_exclude mfn >> $${test}.tmp 2>&1; \
	${TESTCODE}

