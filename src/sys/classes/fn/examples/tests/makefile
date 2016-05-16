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
LOCDIR     = src/sys/classes/fn/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c test7.c test8.c
EXAMPLESF  =
MANSEC     = FN
TESTS      = test1 test2 test3 test4 test5 test6 test7 test8

TESTEXAMPLES_C       = test1.PETSc runtest1_1 test1.rm \
                       test2.PETSc runtest2_1 test2.rm \
                       test3.PETSc runtest3_1 runtest3_2 test3.rm \
                       test4.PETSc runtest4_1 test4.rm \
                       test5.PETSc runtest5_1 runtest5_2 test5.rm \
                       test6.PETSc runtest6_1 runtest6_2 test6.rm \
                       test7.PETSc runtest7_1 runtest7_2 test7.rm \
                       test8.PETSc runtest8_1 runtest8_2 test8.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_SYS_LIB}
	${RM} test1.o

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

test8: test8.o chkopts
	-${CLINKER} -o test8 test8.o ${SLEPC_SYS_LIB}
	${RM} test8.o

#------------------------------------------------------------------------------------

runtest1_1:
	-@${MPIEXEC} -n 1 ./test1 > test1_1.tmp 2>&1; \
	  if (${DIFF} output/test1_1.out test1_1.tmp) then true; \
	  else echo "Possible problem with test1_1, diffs above"; fi; \
	  ${RM} -f test1_1.tmp

runtest2_1:
	-@${MPIEXEC} -n 1 ./test2 > test2_1.tmp 2>&1; \
	  if (${DIFF} output/test2_1.out test2_1.tmp) then true; \
	  else echo "Possible problem with test2_1, diffs above"; fi; \
	  ${RM} -f test2_1.tmp

runtest3_1:
	-@${MPIEXEC} -n 1 ./test3 > test3_1.tmp 2>&1; \
	  if (${DIFF} output/test3_1.out test3_1.tmp) then true; \
	  else echo "Possible problem with test3_1, diffs above"; fi; \
	  ${RM} -f test3_1.tmp

runtest3_2:
	-@${MPIEXEC} -n 1 ./test3 -inplace > test3_1.tmp 2>&1; \
	  if (${DIFF} output/test3_1.out test3_1.tmp) then true; \
	  else echo "Possible problem with test3_1, diffs above"; fi; \
	  ${RM} -f test3_1.tmp

runtest4_1:
	-@${MPIEXEC} -n 1 ./test4 -f1_fn_type exp -f1_fn_scale -2.5 -f2_fn_type rational -f2_fn_rational_numerator -1,1 -f2_fn_rational_denominator 1,-6,4 > test4_1.tmp 2>&1; \
	  if (${DIFF} output/test4_1.out test4_1.tmp) then true; \
	  else echo "Possible problem with test4_1, diffs above"; fi; \
	  ${RM} -f test4_1.tmp

runtest5_1:
	-@${MPIEXEC} -n 1 ./test5 > test5_1.tmp 2>&1; \
	  if (${DIFF} output/test5_1.out test5_1.tmp) then true; \
	  else echo "Possible problem with test5_1, diffs above"; fi; \
	  ${RM} -f test5_1.tmp

runtest5_2:
	-@${MPIEXEC} -n 1 ./test5 -inplace > test5_1.tmp 2>&1; \
	  if (${DIFF} output/test5_1.out test5_1.tmp) then true; \
	  else echo "Possible problem with test5_1, diffs above"; fi; \
	  ${RM} -f test5_1.tmp

runtest6_1:
	-@${MPIEXEC} -n 1 ./test6 > test6_1.tmp 2>&1; \
	  if (${DIFF} output/test6_1.out test6_1.tmp) then true; \
	  else echo "Possible problem with test6_1, diffs above"; fi; \
	  ${RM} -f test6_1.tmp

runtest6_2:
	-@${MPIEXEC} -n 1 ./test6 -inplace > test6_1.tmp 2>&1; \
	  if (${DIFF} output/test6_1.out test6_1.tmp) then true; \
	  else echo "Possible problem with test6_1, diffs above"; fi; \
	  ${RM} -f test6_1.tmp

runtest7_1:
	-@${MPIEXEC} -n 1 ./test7 -tau .13 -eta 2 -n 19 > test7_1.tmp 2>&1; \
	  if (${DIFF} output/test7_1.out test7_1.tmp) then true; \
	  else echo "Possible problem with test7_1, diffs above"; fi; \
	  ${RM} -f test7_1.tmp

runtest7_2:
	-@${MPIEXEC} -n 1 ./test7 -tau .13 -eta 2 -n 19 -inplace > test7_1.tmp 2>&1; \
	  if (${DIFF} output/test7_1.out test7_1.tmp) then true; \
	  else echo "Possible problem with test7_1, diffs above"; fi; \
	  ${RM} -f test7_1.tmp

runtest8_1:
	-@${MPIEXEC} -n 1 ./test8 -tau 0.9 -eta 0.5 -n 10 > test8_1.tmp 2>&1; \
	  if (${DIFF} output/test8_1.out test8_1.tmp) then true; \
	  else echo "Possible problem with test8_1, diffs above"; fi; \
	  ${RM} -f test8_1.tmp

runtest8_2:
	-@${MPIEXEC} -n 1 ./test8 -tau 0.9 -eta 0.5 -n 10 -inplace > test8_1.tmp 2>&1; \
	  if (${DIFF} output/test8_1.out test8_1.tmp) then true; \
	  else echo "Possible problem with test8_1, diffs above"; fi; \
	  ${RM} -f test8_1.tmp

