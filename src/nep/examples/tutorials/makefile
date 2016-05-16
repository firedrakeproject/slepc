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
LOCDIR     = src/nep/examples/tutorials/
EXAMPLESC  = ex20.c ex21.c ex22.c ex27.c
EXAMPLESF  =
MANSEC     = NEP

TESTEXAMPLES_C           = ex21.PETSc runex21_1 ex21.rm \
                           ex22.PETSc runex22_1 ex22.rm \
                           ex27.PETSc runex27_1 runex27_2 ex27.rm
TESTEXAMPLES_C_COMPLEX   = ex22.PETSc runex22_ciss_1 ex22.rm
TESTEXAMPLES_F90         =

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex20: ex20.o chkopts
	-${CLINKER} -o ex20 ex20.o ${SLEPC_NEP_LIB}
	${RM} ex20.o

ex21: ex21.o chkopts
	-${CLINKER} -o ex21 ex21.o ${SLEPC_NEP_LIB}
	${RM} ex21.o

ex22: ex22.o chkopts
	-${CLINKER} -o ex22 ex22.o ${SLEPC_NEP_LIB}
	${RM} ex22.o

ex27: ex27.o chkopts
	-${CLINKER} -o ex27 ex27.o ${SLEPC_NEP_LIB}
	${RM} ex27.o

#------------------------------------------------------------------------------------

runex20_1:
	-@${MPIEXEC} -n 1 ./ex20 > ex20_1.tmp 2>&1; \
	   if (${DIFF} output/ex20_1.out ex20_1.tmp) then true; \
	   else echo "Possible problem with ex20_1, diffs above"; fi; \
	   ${RM} -f ex20_1.tmp

runex21_1:
	-@${MPIEXEC} -n 1 ./ex21 -terse > ex21_1.tmp 2>&1; \
	   if (${DIFF} output/ex21_1.out ex21_1.tmp) then true; \
	   else echo "Possible problem with ex21_1, diffs above"; fi; \
	   ${RM} -f ex21_1.tmp

runex22_1:
	-@${MPIEXEC} -n 1 ./ex22 -terse > ex22_1.tmp 2>&1; \
	   if (${DIFF} output/ex22_1.out ex22_1.tmp) then true; \
	   else echo "Possible problem with ex22_1, diffs above"; fi; \
	   ${RM} -f ex22_1.tmp

runex22_ciss_1:
	-@${MPIEXEC} -n 1 ./ex22 -nep_type ciss -rg_type ellipse -rg_ellipse_center 10 -rg_ellipse_radius 9.5 -nep_ncv 24 -terse > ex22_ciss_1.tmp 2>&1; \
	   if (${DIFF} output/ex22_ciss_1.out ex22_ciss_1.tmp) then true; \
	   else echo "Possible problem with ex22_ciss_1, diffs above"; fi; \
	   ${RM} -f ex22_ciss_1.tmp

runex27_1:
	-@${MPIEXEC} -n 1 ./ex27 -nep_nev 3 -terse > ex27_1.tmp 2>&1; \
	   if (${DIFF} output/ex27_1.out ex27_1.tmp) then true; \
	   else echo "Possible problem with ex27_1, diffs above"; fi; \
	   ${RM} -f ex27_1.tmp

runex27_2:
	-@${MPIEXEC} -n 1 ./ex27 -split 0 -nep_nev 3 -terse > ex27_2.tmp 2>&1; \
	   if (${DIFF} output/ex27_2.out ex27_2.tmp) then true; \
	   else echo "Possible problem with ex27_2, diffs above"; fi; \
	   ${RM} -f ex27_2.tmp

