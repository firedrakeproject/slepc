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
LOCDIR     = src/sys/classes/fn/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c test7.c test8.c test9.c test10.c test11.c test12.c
EXAMPLESF  = test1f.F test7f.F
MANSEC     = FN
TESTS      = test1 test1f test2 test3 test4 test5 test6 test7 test7f test8 test9 test10 test11 test12

TESTEXAMPLES_C           = test1.PETSc runtest1_1 test1.rm \
                           test2.PETSc runtest2_1 test2.rm \
                           test3.PETSc runtest3_1 runtest3_2 runtest3_3 runtest3_4 test3.rm \
                           test4.PETSc runtest4_1 test4.rm \
                           test5.PETSc runtest5_1 runtest5_2 test5.rm \
                           test6.PETSc runtest6_1 runtest6_2 test6.rm \
                           test7.PETSc runtest7_1 runtest7_2 runtest7_3 runtest7_4 test7.rm \
                           test8.PETSc runtest8_1 runtest8_2 test8.rm \
                           test9.PETSc runtest9_1 test9.rm \
                           test10.PETSc runtest10_1 test10.rm \
                           test11.PETSc runtest11_1 runtest11_2 test11.rm \
                           test12.PETSc runtest12_1 test12.rm
TESTEXAMPLES_C_NOTSINGLE = test7.PETSc runtest7_1_sadeghi runtest7_2_sadeghi test7.rm
TESTEXAMPLES_FORTRAN     = test1f.PETSc runtest1f_1 test1f.rm \
                           test7f.PETSc runtest7f_1 test7f.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_SYS_LIB}
	${RM} test1.o

test1f: test1f.o chkopts
	-${FLINKER} -o test1f test1f.o ${SLEPC_SYS_LIB}
	${RM} test1f.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_SYS_LIB}
	${RM} test2.o

test3: test3.o chkopts
	-${CLINKER} -o test3 test3.o ${SLEPC_SYS_LIB}
	${RM} test3.o

test4: test4.o chkopts
	-${CLINKER} -o test4 test4.o ${SLEPC_SYS_LIB}
	${RM} test4.o

test5: test5.o chkopts
	-${CLINKER} -o test5 test5.o ${SLEPC_SYS_LIB}
	${RM} test5.o

test6: test6.o chkopts
	-${CLINKER} -o test6 test6.o ${SLEPC_SYS_LIB}
	${RM} test6.o

test7: test7.o chkopts
	-${CLINKER} -o test7 test7.o ${SLEPC_SYS_LIB}
	${RM} test7.o

test7f: test7f.o chkopts
	-${FLINKER} -o test7f test7f.o ${SLEPC_SYS_LIB}
	${RM} test7f.o

test8: test8.o chkopts
	-${CLINKER} -o test8 test8.o ${SLEPC_SYS_LIB}
	${RM} test8.o

test9: test9.o chkopts
	-${CLINKER} -o test9 test9.o ${SLEPC_SYS_LIB}
	${RM} test9.o

test10: test10.o chkopts
	-${CLINKER} -o test10 test10.o ${SLEPC_SYS_LIB}
	${RM} test10.o

test11: test11.o chkopts
	-${CLINKER} -o test11 test11.o ${SLEPC_SYS_LIB}
	${RM} test11.o

test12: test12.o chkopts
	-${CLINKER} -o test12 test12.o ${SLEPC_SYS_LIB}
	${RM} test12.o

#------------------------------------------------------------------------------------

runtest1_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test1 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest1f_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test1f > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1: runtest2_1_higham runtest2_1_pade
runtest2_1_%:
	-@${SETTEST}; check=test2_1; method=$*; \
	if [ "$$method" = higham ]; then method="-fn_method 0"; \
	elif [ "$$method" = pade ]; then method="-fn_method 1"; fi; \
	${MPIEXEC} -n 1 ./test2 $$method | ${GREP} -v "computing matrix functions" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_1: runtest3_1_higham runtest3_1_pade runtest3_1_subdiagonalpade
runtest3_1_%:
	-@${SETTEST}; check=test3_1; method=$*; \
	if [ "$$method" != subdiagonalpade ] || ${GREP} "\#define PETSC_HAVE_C99_COMPLEX" ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h > /dev/null; then \
	  if [ "$$method" = higham ]; then method="-fn_method 0"; \
	  elif [ "$$method" = pade ]; then method="-fn_method 1"; \
	  elif [ "$$method" = subdiagonalpade ]; then method="-fn_method 2"; fi; \
	  ${MPIEXEC} -n 1 ./test3 $$method | ${GREP} -v "computing matrix functions" > $${test}.tmp 2>&1; \
	  ${TESTCODE}; \
	fi

runtest3_2:
	-@${SETTEST}; check=test3_1; \
	${MPIEXEC} -n 1 ./test3 -inplace | ${GREP} -v "computing matrix functions" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_3: runtest3_3_higham runtest3_3_pade runtest3_3_subdiagonalpade
runtest3_3_%:
	-@${SETTEST}; check=test3_3; method=$*; \
	if [ "$$method" != subdiagonalpade ] || ${GREP} "\#define PETSC_HAVE_C99_COMPLEX" ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h > /dev/null; then \
	  if [ "$$method" = higham ]; then method="-fn_method 0"; \
	  elif [ "$$method" = pade ]; then method="-fn_method 1"; \
	  elif [ "$$method" = subdiagonalpade ]; then method="-fn_method 2"; fi; \
	  ${MPIEXEC} -n 1 ./test3 -fn_scale 0.1 $$method | ${GREP} -v "computing matrix functions" > $${test}.tmp 2>&1; \
	  ${TESTCODE}; \
	fi

runtest3_4: runtest3_4_higham runtest3_4_pade runtest3_4_subdiagonalpade
runtest3_4_%:
	-@${SETTEST}; check=test3_4; method=$*; \
	if [ "$$method" != subdiagonalpade ] || ${GREP} "\#define PETSC_HAVE_C99_COMPLEX" ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h > /dev/null; then \
	  if [ "$$method" = higham ]; then method="-fn_method 0"; \
	  elif [ "$$method" = pade ]; then method="-fn_method 1"; \
	  elif [ "$$method" = subdiagonalpade ]; then method="-fn_method 2"; fi; \
	  ${MPIEXEC} -n 1 ./test3 -n 200 -fn_scale 0.6,1.5 $$method | ${GREP} -v "computing matrix functions" > $${test}.tmp 2>&1; \
	  ${TESTCODE}; \
	fi

runtest4_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test4 -f1_fn_type exp -f1_fn_scale -2.5 -f2_fn_type rational -f2_fn_rational_numerator -1,1 -f2_fn_rational_denominator 1,-6,4 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test5 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest5_2:
	-@${SETTEST}; check=test5_1; \
	${MPIEXEC} -n 1 ./test5 -inplace > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test6 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest6_2:
	-@${SETTEST}; check=test6_1; \
	${MPIEXEC} -n 1 ./test6 -inplace > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_1: runtest7_1_schur runtest7_1_dbp runtest7_1_ns
runtest7_1_%:
	-@${SETTEST}; check=test7_1; method=$*; \
	if [ "$$method" = schur ]; then method="-fn_method 0"; \
	elif [ "$$method" = dbp ]; then method="-fn_method 1"; \
	elif [ "$$method" = ns ]; then method="-fn_method 2"; \
	elif [ "$$method" = sadeghi ]; then method="-fn_method 3"; fi; \
	${MPIEXEC} -n 1 ./test7 -fn_scale .05,2 -n 100 $$method | ${GREP} -v "computing matrix functions" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_2: runtest7_2_schur runtest7_2_dbp runtest7_2_ns
runtest7_2_%:
	-@${SETTEST}; check=test7_1; method=$*; \
	if [ "$$method" = schur ]; then method="-fn_method 0"; \
	elif [ "$$method" = dbp ]; then method="-fn_method 1"; \
	elif [ "$$method" = ns ]; then method="-fn_method 2"; \
	elif [ "$$method" = sadeghi ]; then method="-fn_method 3"; fi; \
	${MPIEXEC} -n 1 ./test7 -fn_scale .05,2 -n 100 -inplace $$method | ${GREP} -v "computing matrix functions" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_3:
	-@${SETTEST}; check=test7_1; \
	${MPIEXEC} -n 3 ./test7 -fn_scale .05,2 -n 100 -fn_parallel synchronized | ${GREP} -v "computing matrix functions"  | ${GREP} -v "SYNCHRONIZED" | ${SED} -e "s/3 MPI/1 MPI/g" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7_4:
	-@${SETTEST}; check=test7_1; \
	${MPIEXEC} -n 3 ./test7 -fn_scale .05,2 -n 100 -inplace -fn_parallel synchronized | ${GREP} -v "computing matrix functions" | ${GREP} -v "SYNCHRONIZED" | ${SED} -e "s/3 MPI/1 MPI/g" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest7f_1: runtest7f_1_schur runtest7f_1_dbp runtest7f_1_ns runtest7f_1_sadeghi
runtest7f_1_%:
	-@${SETTEST}; check=test7f_1; method=$*; \
	if [ "$$method" = schur ]; then method="-fn_method 0"; \
	elif [ "$$method" = dbp ]; then method="-fn_method 1"; \
	elif [ "$$method" = ns ]; then method="-fn_method 2"; \
	elif [ "$$method" = sadeghi ]; then method="-fn_method 3"; fi; \
	${MPIEXEC} -n 1 ./test7f -fn_scale .13,2 -n 19 $$method | ${GREP} -v "computing matrix functions" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_1: runtest8_1_schur runtest8_1_dbp runtest8_1_ns
runtest8_1_%:
	-@${SETTEST}; check=test8_1; method=$*; \
	if [ "$$method" = schur ]; then method="-fn_method 0"; \
	elif [ "$$method" = dbp ]; then method="-fn_method 1"; \
	elif [ "$$method" = ns ]; then method="-fn_method 2"; fi; \
	${MPIEXEC} -n 1 ./test8 -fn_scale 0.9,0.5 -n 10 $$method | ${GREP} -v "computing matrix functions" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest8_2: runtest8_2_schur runtest8_2_dbp runtest8_2_ns
runtest8_2_%:
	-@${SETTEST}; check=test8_1; method=$*; \
	if [ "$$method" = schur ]; then method="-fn_method 0"; \
	elif [ "$$method" = dbp ]; then method="-fn_method 1"; \
	elif [ "$$method" = ns ]; then method="-fn_method 2"; fi; \
	${MPIEXEC} -n 1 ./test8 -fn_scale 0.9,0.5 -n 10 -inplace $$method | ${GREP} -v "computing matrix functions" > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest9_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test9 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest10_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test10 -fn_phi_index 3 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_1:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./test11 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest11_2:
	-@${SETTEST}; check=test11_1; \
	${MPIEXEC} -n 1 ./test11 -inplace > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest12_1: runtest12_1_exp runtest12_1_sqrt runtest12_1_rational
runtest12_1_%:
	-@${SETTEST}; check=test12_1; fun=$*; \
	if [ "$$fun" = rational ]; then fun="rational -fn_rational_numerator 2,-1.5 -fn_rational_denominator 1,0.8"; fi; \
	${MPIEXEC} -n 1 ./test12 -fn_type $$fun > $${test}.tmp 2>&1; \
	${TESTCODE}

