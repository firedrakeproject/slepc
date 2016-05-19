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
LOCDIR     = src/sys/classes/ds/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c test4.c test5.c test6.c test7.c test8.c test9.c \
             test12.c test13.c test15.c
EXAMPLESF  = test14f.F
MANSEC     = DS
TESTS      = test1 test2 test3 test4 test5 test6 test7 test8 test9 test12 test13 \
             test14f test15

TESTEXAMPLES_C           = test1.PETSc runtest1_1 test1.rm \
                           test2.PETSc runtest2_1 test2.rm \
                           test3.PETSc runtest3_1 test3.rm \
                           test4.PETSc runtest4_1 test4.rm \
                           test5.PETSc runtest5_1 test5.rm \
                           test6.PETSc runtest6_1 test6.rm \
                           test7.PETSc runtest7_1 test7.rm \
                           test8.PETSc runtest8_1 test8.rm \
                           test9.PETSc runtest9_1 test9.rm \
                           test12.PETSc runtest12_1 test12.rm \
                           test15.PETSc runtest15_1 test15.rm
TESTEXAMPLES_C_NOCOMPLEX = test13.PETSc runtest13_1 test13.rm
TESTEXAMPLES_FORTRAN     = test14f.PETSc runtest14f_1 test14f.rm

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

test9: test9.o chkopts
	-${CLINKER} -o test9 test9.o ${SLEPC_SYS_LIB}
	${RM} test9.o

test12: test12.o chkopts
	-${CLINKER} -o test12 test12.o ${SLEPC_SYS_LIB}
	${RM} test12.o

test13: test13.o chkopts
	-${CLINKER} -o test13 test13.o ${SLEPC_SYS_LIB}
	${RM} test13.o

test14f: test14f.o chkopts
	-${FLINKER} -o test14f test14f.o ${SLEPC_SYS_LIB}
	${RM} test14f.o

test15: test15.o chkopts
	-${CLINKER} -o test15 test15.o ${SLEPC_SYS_LIB}
	${RM} test15.o

#------------------------------------------------------------------------------------

runtest1_1:
	-@${MPIEXEC} -n 1 ./test1 > test1_1.tmp 2>&1; \
	  if (${DIFF} output/test1_1.out test1_1.tmp) then true; \
	  else echo "Possible problem with test1_1, diffs above"; fi; \
	  ${RM} -f test1_1.tmp

runtest2_1:
	-@${MPIEXEC} -n 1 ./test2 -n 12 > test2_1.tmp 2>&1; \
	  if (${DIFF} output/test2_1.out test2_1.tmp) then true; \
	  else echo "Possible problem with test2_1, diffs above"; fi; \
	  ${RM} -f test2_1.tmp

runtest3_1:
	-@${MPIEXEC} -n 1 ./test3 > test3_1.tmp 2>&1; \
	  if (${DIFF} output/test3_1.out test3_1.tmp) then true; \
	  else echo "Possible problem with test3_1, diffs above"; fi; \
	  ${RM} -f test3_1.tmp

runtest4_1:
	-@${MPIEXEC} -n 1 ./test4 > test4_1.tmp 2>&1; \
	  if (${DIFF} output/test4_1.out test4_1.tmp) then true; \
	  else echo "Possible problem with test4_1, diffs above"; fi; \
	  ${RM} -f test4_1.tmp

runtest5_1:
	-@${MPIEXEC} -n 1 ./test5 > test5_1.tmp 2>&1; \
	  if (${DIFF} output/test5_1.out test5_1.tmp) then true; \
	  else echo "Possible problem with test5_1, diffs above"; fi; \
	  ${RM} -f test5_1.tmp

runtest6_1:
	-@${MPIEXEC} -n 1 ./test6 > test6_1.tmp 2>&1; \
	  if (${DIFF} output/test6_1.out test6_1.tmp) then true; \
	  else echo "Possible problem with test6_1, diffs above"; fi; \
	  ${RM} -f test6_1.tmp

runtest7_1:
	-@${MPIEXEC} -n 1 ./test7 > test7_1.tmp 2>&1; \
	  if (${DIFF} output/test7_1.out test7_1.tmp) then true; \
	  else echo "Possible problem with test7_1, diffs above"; fi; \
	  ${RM} -f test7_1.tmp

runtest8_1:
	-@${MPIEXEC} -n 1 ./test8 > test8_1.tmp 2>&1; \
	  if (${DIFF} output/test8_1.out test8_1.tmp) then true; \
	  else echo "Possible problem with test8_1, diffs above"; fi; \
	  ${RM} -f test8_1.tmp

runtest9_1:
	-@${MPIEXEC} -n 1 ./test9 > test9_1.tmp 2>&1; \
	  if (${DIFF} output/test9_1.out test9_1.tmp) then true; \
	  else echo "Possible problem with test9_1, diffs above"; fi; \
	  ${RM} -f test9_1.tmp

runtest12_1:
	-@${MPIEXEC} -n 1 ./test12 > test12_1.tmp 2>&1; \
	  if (${DIFF} output/test12_1.out test12_1.tmp) then true; \
	  else echo "Possible problem with test12_1, diffs above"; fi; \
	  ${RM} -f test12_1.tmp

runtest13_1:
	-@${MPIEXEC} -n 1 ./test13 > test13_1.tmp 2>&1; \
	  if (${DIFF} output/test13_1.out test13_1.tmp) then true; \
	  else echo "Possible problem with test13_1, diffs above"; fi; \
	  ${RM} -f test13_1.tmp

runtest14f_1:
	-@${MPIEXEC} -n 1 ./test14f > test14f_1.tmp 2>&1; \
	  if (${DIFF} output/test14f_1.out test14f_1.tmp) then true; \
	  else echo "Possible problem with test14f_1, diffs above"; fi; \
	  ${RM} -f test14f_1.tmp

runtest15_1:
	-@${MPIEXEC} -n 1 ./test15 > test15_1.tmp 2>&1; \
	  if (${DIFF} output/test15_1.out test15_1.tmp) then true; \
	  else echo "Possible problem with test15_1, diffs above"; fi; \
	  ${RM} -f test15_1.tmp

