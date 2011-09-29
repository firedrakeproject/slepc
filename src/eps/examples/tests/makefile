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
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c
EXAMPLESF  = test7f.F
MANSEC     = EPS
TESTS      = test1 test2 test3 test4 test5 test6 test7f

TESTEXAMPLES_C       = test1.PETSc runtest1_1 test1.rm \
                       test2.PETSc runtest2_1 test2.rm \
                       test3.PETSc runtest3_1 test3.rm \
                       test4.PETSc runtest4_1 test4.rm \
                       test5.PETSc runtest5_1 test5.rm \
                       test6.PETSc runtest6_1 test6.rm 
TESTEXAMPLES_FORTRAN = test7f.PETSc runtest7f_1 test7f.rm
TESTEXAMPLES_BLOPEX  = test5.PETSc runtest5_blopex test5.rm 

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

test4: test4.o chkopts
	-${CLINKER} -o test4 test4.o ${SLEPC_LIB}
	${RM} test4.o

test5: test5.o chkopts
	-${CLINKER} -o test5 test5.o ${SLEPC_LIB}
	${RM} test5.o

test6: test6.o chkopts
	-${CLINKER} -o test6 test6.o ${SLEPC_LIB}
	${RM} test6.o

test7f: test7f.o chkopts
	-${FLINKER} -o test7f test7f.o ${SLEPC_LIB}
	${RM} test7f.o

#------------------------------------------------------------------------------------
EPSALL = krylovschur arnoldi lanczos gd jd
EPSNS  = krylovschur arnoldi gd jd
runtest1_1:
	-@touch test1_1.tmp; \
	for eps in ${EPSALL}; do \
	   echo "eps type $$eps" >> test1_1.tmp; \
	   if [ "$$eps" == "lanczos" ]; then EXTRA="-eps_lanczos_reorthog full"; else EXTRA=""; fi; \
	   ${MPIEXEC} -np 1 ./test1 -eps_type $$eps -eps_nev 4 $$EXTRA -eps_terse >> test1_1.tmp 2>&1; \
	done; \
	if (${DIFF} output/test1_1.out test1_1.tmp) then true; \
	else echo "Possible problem with test1_1, diffs above"; fi; \
	${RM} -f test1_1.tmp

runtest2_1:
	-@touch test2_1.tmp; \
	for eps in ${EPSALL}; do \
	   echo "eps type $$eps" >> test2_1.tmp; \
	   ${MPIEXEC} -np 1 ./test2 -eps_type $$eps -eps_nev 4 -eps_terse >> test2_1.tmp 2>&1;    \
	done; \
	if (${DIFF} output/test2_1.out test2_1.tmp) then true; \
	else echo "Possible problem with test2_1, diffs above"; fi; \
	${RM} -f test2_1.tmp

runtest3_1:
	-@touch test3_1.tmp; \
	for eps in ${EPSALL}; do \
	   echo "eps type $$eps" >> test3_1.tmp; \
	   ${MPIEXEC} -np 1 ./test3 -eps_type $$eps -eps_nev 4 -eps_terse >> test3_1.tmp 2>&1;    \
	done; \
	if (${DIFF} output/test3_1.out test3_1.tmp) then true; \
	else echo "Possible problem with test3_1, diffs above"; fi; \
	${RM} -f test3_1.tmp

runtest4_1:
	-@touch test4_1.tmp; \
	for eps in ${EPSALL}; do \
	   echo "eps type $$eps" >> test4_1.tmp; \
	   ${MPIEXEC} -np 1 ./test4 -type $$eps -eps_terse >> test4_1.tmp 2>&1;    \
	done; \
	if (${DIFF} output/test4_1.out test4_1.tmp) then true; \
	else echo "Possible problem with test4_1, diffs above"; fi; \
	${RM} -f test4_1.tmp

runtest5_1:
	-@touch test5_1.tmp; \
	for eps in ${EPSNS}; do \
	   echo "eps type $$eps" >> test5_1.tmp; \
	   ${MPIEXEC} -np 1 ./test5 -eps_type $$eps -eps_nev 4 -eps_terse >> test5_1.tmp 2>&1;    \
	done; \
	if (${GREP} USE_COMPLEX ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h > /dev/null 2>&1) then \
	  if (${DIFF} output/test5_1_complex.out test5_1.tmp) then true; \
	  else echo "Possible problem with test5_1, diffs above"; fi; \
	else \
	  if (${DIFF} output/test5_1.out test5_1.tmp) then true; \
	  else echo "Possible problem with test5_1, diffs above"; fi; \
	fi; \
	${RM} -f test5_1.tmp

runtest5_blopex:
	-@${MPIEXEC} -np 1 ./test5 -symm -eps_type blopex -eps_nev 4 -eps_terse > test5_blopex.tmp 2>&1; \
	if (${DIFF} output/test5_blopex.out test5_blopex.tmp) then true; \
	else echo "Possible problem with test5_blopex, diffs above"; fi; \
	${RM} -f test5_blopex.tmp;

testtest5_blopex: test5.PETSc
	@if [ "${PETSC_WITH_BATCH}" != "" -o "${MPIEXEC}" = "/bin/false" ]; then \
           echo "Skipping BLOPEX test"; \
	elif [ -f test5 ]; then \
           ${MPIEXEC} -np 1 ./test5 -symm -eps_type blopex -eps_nev 4 -eps_terse > test5_blopex.tmp 2>&1; \
	   if (${DIFF} output/test5_blopex.out test5_blopex.tmp > /dev/null 2>&1) then \
           echo "BLOPEX example src/eps/examples/tests/test5 run successfully with 1 MPI process"; \
	   else echo "Possible error running BLOPEX src/eps/examples/tests/test5 with 1 MPI process"; \
           cat test5_blopex.tmp; fi; \
           ${RM} -f test5_blopex.tmp; \
           ${MAKE} SLEPC_ARCH=${SLEPC_ARCH} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} test5.rm ; fi

runtest6_1:
	-@touch test6_1.tmp; \
	for eps in ${EPSNS}; do \
	   echo "eps type $$eps" >> test6_1.tmp; \
	   ${MPIEXEC} -np 1 ./test6 -eps_type $$eps -eps_nev 4 -eps_terse >> test6_1.tmp 2>&1;    \
	done; \
	if (${DIFF} output/test6_1.out test6_1.tmp) then true; \
	else echo "Possible problem with test6_1, diffs above"; fi; \
	${RM} -f test6_1.tmp

runtest7f_1:
	-@${MPIEXEC} -np 1 ./test7f -eps_nev 4 -eps_terse > test7f_1.tmp 2>&1; \
	   if (${DIFF} output/test7f_1.out test7f_1.tmp) then true; \
	   else echo "Possible problem with test7f_1, diffs above"; fi; \
	   ${RM} -f test7f_1.tmp

testtest7f: test7f.PETSc
	@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
	   echo "Running with batch filesystem; to test run src/eps/examples/tests/test7f " ; \
	   echo "with your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/eps/examples/tests/test7f manually"; \
	elif [ -f test7f ]; then \
           ${MPIEXEC} -np 1 ./test7f -eps_nev 4 -eps_terse > test7f_1.tmp 2>&1; \
	   if (${DIFF} output/test7f_1.out test7f_1.tmp > /dev/null 2>&1) then \
           echo "Fortran example src/eps/examples/tests/test7f run successfully with 1 MPI process"; \
	   else echo "Possible error running Fortran src/eps/examples/tests/test7f with 1 MPI process"; \
           cat test7f_1.tmp; fi; \
           ${RM} -f test7f_1.tmp; \
           ${MAKE} SLEPC_ARCH=${SLEPC_ARCH} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} test7f.rm ; fi


