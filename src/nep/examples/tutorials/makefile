#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain
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
LOCDIR     = src/nep/examples/tutorials/
EXAMPLESC  = ex20.c ex21.c ex22.c
EXAMPLESF  =
MANSEC     = NEP

TESTEXAMPLES_C           = ex20.PETSc runex20_1 ex20.rm \
                           ex21.PETSc runex21_1 ex21.rm \
                           ex22.PETSc runex22_1 ex22.rm
TESTEXAMPLES_C_NOCOMPLEX =
TESTEXAMPLES_F90         =

include ${SLEPC_DIR}/conf/slepc_common

ex20: ex20.o chkopts
	-${CLINKER} -o ex20 ex20.o ${SLEPC_NEP_LIB}
	${RM} ex20.o

ex21: ex21.o chkopts
	-${CLINKER} -o ex21 ex21.o ${SLEPC_NEP_LIB}
	${RM} ex21.o

ex22: ex22.o chkopts
	-${CLINKER} -o ex22 ex22.o ${SLEPC_NEP_LIB}
	${RM} ex22.o

#------------------------------------------------------------------------------------

runex20_1:
	-@${MPIEXEC} -np 1 ./ex20 > ex20_1.tmp 2>&1; \
	   if (${DIFF} output/ex20_1.out ex20_1.tmp) then true; \
	   else echo "Possible problem with ex20_1, diffs above"; fi; \
	   ${RM} -f ex20_1.tmp

runex21_1:
	-@${MPIEXEC} -np 1 ./ex21 > ex21_1.tmp 2>&1; \
	   if (${DIFF} output/ex21_1.out ex21_1.tmp) then true; \
	   else echo "Possible problem with ex21_1, diffs above"; fi; \
	   ${RM} -f ex21_1.tmp

runex22_1:
	-@${MPIEXEC} -np 1 ./ex22 > ex22_1.tmp 2>&1; \
	   if (${DIFF} output/ex22_1.out ex22_1.tmp) then true; \
	   else echo "Possible problem with ex22_1, diffs above"; fi; \
	   ${RM} -f ex22_1.tmp

