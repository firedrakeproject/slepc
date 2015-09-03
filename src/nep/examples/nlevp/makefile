#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain
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
EXAMPLESC  = loaded_string.c
EXAMPLESF  = 
MANSEC     = NEP
TESTS      = loaded_string

TESTEXAMPLES_C = loaded_string.PETSc runloaded_string loaded_string.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

loaded_string: loaded_string.o chkopts
	-${CLINKER} -o loaded_string loaded_string.o ${SLEPC_NEP_LIB}
	${RM} loaded_string.o

#------------------------------------------------------------------------------------
DATAPATH = ${SLEPC_DIR}/share/slepc/datafiles/matrices

runloaded_string:
	-@${MPIEXEC} -np 1 ./loaded_string -nep_type rii -n 100 -terse > loaded_string_1.tmp 2>&1; \
	   if (${DIFF} output/loaded_string_1.out loaded_string_1.tmp) then true; \
	   else echo "Possible problem with loaded_string_1, diffs above"; fi; \
	   ${RM} -f loaded_string_1.tmp

