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
LOCDIR     = src/nep/examples/nlevp/
EXAMPLESC  = loaded_string.c gun.c
EXAMPLESF  = 
MANSEC     = NEP
TESTS      = loaded_string gun

TESTEXAMPLES_C = loaded_string.PETSc runloaded_string loaded_string.rm
TESTEXAMPLES_DATAFILESPATH_COMPLEX = gun.PETSc rungun gun.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

loaded_string: loaded_string.o chkopts
	-${CLINKER} -o loaded_string loaded_string.o ${SLEPC_NEP_LIB}
	${RM} loaded_string.o

gun: gun.o chkopts
	-${CLINKER} -o gun gun.o ${SLEPC_NEP_LIB}
	${RM} gun.o

#------------------------------------------------------------------------------------
runloaded_string:
	-@${MPIEXEC} -n 1 ./loaded_string -nep_type rii -n 100 -terse > loaded_string_1.tmp 2>&1; \
	   if (${DIFF} output/loaded_string_1.out loaded_string_1.tmp) then true; \
	   else echo "Possible problem with loaded_string_1, diffs above"; fi; \
	   ${RM} -f loaded_string_1.tmp

rungun:
	-@${MPIEXEC} -n 1 ./gun -K $(DATAFILESPATH)/matrices/complex/gun_K.petsc -M $(DATAFILESPATH)/matrices/complex/gun_M.petsc -W1 $(DATAFILESPATH)/matrices/complex/gun_W1.petsc -W2 $(DATAFILESPATH)/matrices/complex/gun_W2.petsc -nep_type nleigs -rg_type interval -rg_interval_endpoints 12500,120500,-1,25000 -nep_target 65000 -nep_nev 4 -nep_tol 1e-12 -terse > gun_1.tmp 2>&1; \
	   if (${DIFF} output/gun_1.out gun_1.tmp) then true; \
	   else echo "Possible problem with gun_1, diffs above"; fi; \
	   ${RM} -f gun_1.tmp;

