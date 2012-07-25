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
LOCDIR     = src/eps/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c \
             test8.c test9.c test10.c test11.c test12.c test13.c
EXAMPLESF  = test7f.F
MANSEC     = EPS
TESTS      = test1 test2 test3 test4 test5 test6 test7f test8 test9 test10 test11 test12 test13

TESTEXAMPLES_C       = test1.PETSc runtest1_1 runtest1_2 test1.rm \
                       test2.PETSc runtest2_1 test2.rm \
                       test3.PETSc runtest3_1 test3.rm \
                       test4.PETSc runtest4_1 test4.rm \
                       test6.PETSc runtest6_1 test6.rm \
                       test8.PETSc runtest8_1 test8.rm \
                       test9.PETSc runtest9_1 test9.rm \
                       test10.PETSc runtest10_1 test10.rm \
                       test11.PETSc runtest11_1 test11.rm \
                       test12.PETSc runtest12_1 test12.rm \
                       test13.PETSc runtest13_1 test13.rm
TESTEXAMPLES_C_NOF128= test5.PETSc runtest5_1 test5.rm
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

test8: test8.o chkopts
	-${CLINKER} -o test8 test8.o ${SLEPC_LIB}
	${RM} test8.o

test9: test9.o chkopts
	-${CLINKER} -o test9 test9.o ${SLEPC_LIB}
	${RM} test9.o

test10: test10.o chkopts
	-${CLINKER} -o test10 test10.o ${SLEPC_LIB}
	${RM} test10.o

test11: test11.o chkopts
	-${CLINKER} -o test11 test11.o ${SLEPC_LIB}
	${RM} test11.o

test12: test12.o chkopts
	-${CLINKER} -o test12 test12.o ${SLEPC_LIB}
	${RM} test12.o

test13: test13.o chkopts
	-${CLINKER} -o test13 test13.o ${SLEPC_LIB}
	${RM} test13.o

#------------------------------------------------------------------------------------
EPSALL = krylovschur arnoldi lanczos gd jd
EPSNS  = krylovschur arnoldi gd jd
EPSAR  = gd jd

TESTCODE = \
	[ x${SAVE_OUTPUT} = xyes ] && cp $${test}.tmp output/$${test}.out; \
	${DIFF} output/$${test}.out $${test}.tmp || \
	echo "Possible problem with $${test}, diffs above"; \
	${RM} -f $${test}.tmp

runtest1_1:
	-@test=test1_1; \
	for eps in ${EPSALL}; do \
	   echo "eps type $$eps"; \
	   if [ "$$eps" = "lanczos" ]; then EXTRA="-eps_lanczos_reorthog full"; else EXTRA=""; fi; \
	   ${MPIEXEC} -np 1 ./test1 -eps_type $$eps -eps_nev 4 $$EXTRA -eps_terse 2>&1; \
	done > $${test}.tmp; \
	${TESTCODE}

runtest1_2:
	-@test=test1_2; \
	${MPIEXEC} -np 1 ./test1  -eps_interval .1,1.1 -st_type sinvert -st_ksp_type preonly -st_pc_type cholesky -eps_terse 2>&1 >  $${test}.tmp; \
	${TESTCODE}

runtest2_1:
	-@test=test2_1; \
	for eps in ${EPSALL}; do \
	   echo "eps type $$eps"; \
	   if [ $$eps = arnoldi -o $$eps = lanczos ]; then EXTRA="-eps_ncv 15"; else EXTRA=""; fi; \
	   ${MPIEXEC} -np 1 ./test2 -eps_type $$eps -eps_nev 4 -eps_terse $$EXTRA 2>&1;    \
	done > $${test}.tmp; \
	${TESTCODE}

runtest3_1:
	-@test=test3_1; \
	for eps in ${EPSALL}; do \
	   echo "eps type $$eps"; \
	   ${MPIEXEC} -np 1 ./test3 -eps_type $$eps -eps_nev 4 -eps_terse 2>&1;    \
	done > $${test}.tmp; \
	${TESTCODE}

runtest4_1:
	-@test=test4_1; \
	for eps in ${EPSALL}; do \
	   echo "eps type $$eps"; \
	   ${MPIEXEC} -np 1 ./test4 -type $$eps -eps_terse 2>&1;    \
	done > $${test}.tmp; \
	${TESTCODE}

runtest5_1:
	-@test=test5_1; \
	for eps in ${EPSNS}; do \
	   echo "eps type $$eps" >> test5_1.tmp; \
	   if [ $$eps = gd ]; then EXTRA="-eps_ncv 7 -eps_gd_minv 2"; \
	   elif [ $$eps = jd ]; then EXTRA="-eps_ncv 7 -eps_jd_minv 2"; else EXTRA=""; fi; \
	   ${MPIEXEC} -np 1 ./test5 -eps_type $$eps -eps_nev 4 -eps_terse $$EXTRA >> test5_1.tmp 2>&1;    \
	done; \
	if (${GREP} USE_COMPLEX ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h > /dev/null 2>&1) then \
	  [ x${SAVE_OUTPUT} = xyes ] && cp test5_1.tmp output/test5_1_complex.out; \
	  if (${DIFF} output/test5_1_complex.out test5_1.tmp) then true; \
	  else echo "Possible problem with test5_1, diffs above"; fi; \
	else \
	  [ x${SAVE_OUTPUT} = xyes ] && cp test5_1.tmp output/test5_1.out; \
	  if (${DIFF} output/test5_1.out test5_1.tmp) then true; \
	  else echo "Possible problem with test5_1, diffs above"; fi; \
	fi; \
	${RM} -f test5_1.tmp;

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
	-@test=test6_1; \
	for eps in ${EPSNS}; do \
	   echo "eps type $$eps"; \
	   ${MPIEXEC} -np 1 ./test6 -eps_type $$eps -eps_nev 4 -eps_terse 2>&1;    \
	done > $${test}.tmp; \
	${TESTCODE}

runtest7f_1:
	-@test=test7f_1; \
	${MPIEXEC} -np 1 ./test7f -eps_nev 4 -eps_terse > $${test}.tmp 2>&1; \
	${TESTCODE}

testtest7f: test7f.PETSc
	@ok=0; if [ "${PETSC_WITH_BATCH}" != "" ]; then \
	   echo "Running with batch filesystem; to test run src/eps/examples/tests/test7f " ; \
	   echo "with your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/eps/examples/tests/test7f manually"; \
	elif [ -f test7f ]; then \
           ${MPIEXEC} -np 1 ./test7f -eps_nev 4 -eps_terse > test7f_1.tmp 2>&1; \
	   if (${DIFF} output/test7f_1.out test7f_1.tmp > /dev/null 2>&1) then \
           echo "Fortran example src/eps/examples/tests/test7f run successfully with 1 MPI process"; \
	   else echo "Possible error running Fortran src/eps/examples/tests/test7f with 1 MPI process"; \
           cat test7f_1.tmp; ok=1; fi; \
           ${RM} -f test7f_1.tmp; \
           ${MAKE} SLEPC_ARCH=${SLEPC_ARCH} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} test7f.rm ; \
	else ok=1; fi; \
	exit $$ok

runtest8_1:
	-@test=test8_1; \
	for eps in ${EPSALL}; do \
	   echo "eps type $$eps"; \
	   ${MPIEXEC} -np 1 ./test8 -eps_type $$eps -eps_nev 4 -eps_ncv 24 -eps_terse 2>&1; \
	done > $${test}.tmp; \
	${TESTCODE}

runtest9_1:
	-@test=test9_1; \
	for eps in ${EPSNS}; do \
	   echo "eps type $$eps"; \
	   if [ $$eps = jd ]; then EXTRA="-st_ksp_type gmres"; else EXTRA=""; fi; \
	   ${MPIEXEC} -np 1 ./test9 -eps_type $$eps -eps_nev 4 -eps_terse $$EXTRA 2>&1; \
	done > $${test}.tmp;\
	${TESTCODE}

runtest10_1:
	-@test=test10_1; \
	for eps in ${EPSALL}; do \
	   echo "eps type $$eps"; \
	   ${MPIEXEC} -np 1 ./test10 -eps_type $$eps -eps_nev 4 -m 11 -eps_terse 2>&1; \
	done > $${test}.tmp;\
	${TESTCODE}

testtest10: test10.PETSc
	@ok=0; if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/eps/examples/tests/test10" ; \
           echo "with your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/eps/examples/tests/test10 manually"; \
	elif [ -f test10 ]; then \
           ${MPIEXEC} -np 1 ./test10 -eps_nev 4 -m 11 -eps_largest_magnitude -eps_terse > test10_1.tmp 2>&1; \
	   if (${DIFF} output/test10_1_ks.out test10_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/eps/examples/tests/test10 run successfully with 1 MPI process"; \
	   else echo "Possible error running C/C++ src/eps/examples/tests/test10 with 1 MPI process"; \
           cat test10_1.tmp; ok=1; fi; \
	   if [ "${MPIEXEC}" != "${PETSC_DIR}/bin/mpiexec.uni" ]; then \
           ${MPIEXEC} -np 2 ./test10 -eps_nev 4 -m 11 -eps_largest_magnitude -eps_terse > test10_1.tmp 2>&1; \
	   if (${DIFF} output/test10_1_ks.out test10_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/eps/examples/tests/test10 run successfully with 2 MPI process"; \
	   else echo "Possible error running C/C++ src/eps/examples/tests/test10 with 2 MPI process"; \
           cat test10_1.tmp; ok=1; fi; fi; \
           ${RM} -f test10_1.tmp; \
           ${MAKE} SLEPC_ARCH=${SLEPC_ARCH} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} test10.rm; \
	else ok=1; fi; \
	exit $$ok

runtest11_1:
	-@test=test11_1; \
	for eps in ${EPSNS}; do \
	   echo "eps type $$eps"; \
	   if [ $$eps = krylovschur -o $$eps = arnoldi ]; then EXTRA="-st_type sinvert"; else EXTRA=""; fi; \
	   ${MPIEXEC} -np 1 ./test11 -eps_type $$eps -eps_nev 4 -eps_terse $$EXTRA 2>&1; \
	done > $${test}.tmp;\
	${TESTCODE}

runtest12_1:
	-@test=test12_1; \
	for eps in ${EPSNS}; do \
	   echo "eps type $$eps"; \
	   ${MPIEXEC} -np 1 ./test12 -eps_type $$eps -eps_nev 4 -eps_terse 2>&1;    \
	done > $${test}.tmp; \
	${TESTCODE}

runtest13_1:
	-@test=test13_1; \
	for eps in ${EPSAR}; do \
	   echo "eps type $$eps"; \
	   ${MPIEXEC} -np 1 ./test13 -eps_type $$eps -eps_max_it 5000 -eps_terse 2>&1;    \
	done > $${test}.tmp; \
	${TESTCODE}

runtest13_2:
	-@test=test13_2; \
	for eps in ${EPSAR}; do \
	   echo "eps type $$eps"; \
	   ${MPIEXEC} -np 1 ./test13 -eps_type $$eps -eps_non_hermitian -eps_max_it 5000 -eps_terse 2>&1;    \
	done > $${test}.tmp; \
	${TESTCODE}
