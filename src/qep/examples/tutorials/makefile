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
LOCDIR     = src/qep/examples/tutorials/
EXAMPLESC  = ex16.c ex17.c
EXAMPLESF  = ex16f90.F90
MANSEC     = QEP

TESTEXAMPLES_C           = ex16.PETSc runex16_1 ex16.rm
TESTEXAMPLES_C_NOCOMPLEX = ex17.PETSc runex17_1 ex17.rm
TESTEXAMPLES_F90         = ex16f90.PETSc runex16f90_1 ex16f90.rm

include ${SLEPC_DIR}/conf/slepc_common

ex16: ex16.o chkopts
	-${CLINKER} -o ex16 ex16.o ${SLEPC_LIB}
	${RM} ex16.o

ex16f90: ex16f90.o chkopts
	-${FLINKER} -o ex16f90 ex16f90.o ${SLEPC_LIB}
	${RM} ex16f90.o

ex17: ex17.o chkopts
	-${CLINKER} -o ex17 ex17.o ${SLEPC_LIB}
	${RM} ex17.o

#------------------------------------------------------------------------------------
DATAPATH = ${SLEPC_DIR}/share/slepc/datafiles/matrices

runex16_1:
	-@${MPIEXEC} -np 1 ./ex16 -qep_nev 4 -qep_ncv 20 -qep_terse > ex16_1.tmp 2>&1; \
	   if (${DIFF} output/ex16_1.out ex16_1.tmp) then true; \
	   else echo "Possible problem with ex16_1, diffs above"; fi; \
	   ${RM} -f ex16_1.tmp

runex16f90_1:
	-@${MPIEXEC} -np 1 ./ex16f90 -qep_nev 4 -qep_ncv 20 -qep_terse > ex16f90_1.tmp 2>&1; \
	   if (${DIFF} output/ex16f90_1.out ex16f90_1.tmp) then true; \
	   else echo "Possible problem with ex16f90_1, diffs above"; fi; \
	   ${RM} -f ex16f90_1.tmp

runex17_1:
	-@${MPIEXEC} -np 1 ./ex17 -M ${DATAPATH}/speaker107m.petsc -C ${DATAPATH}/speaker107c.petsc -K ${DATAPATH}/speaker107k.petsc -qep_linear_explicitmatrix -qep_nev 4 -qep_ncv 20 -qep_tol 1e-9 -qep_terse > ex17_1.tmp 2>&1; \
	   if (${DIFF} output/ex17_1.out ex17_1.tmp) then true; \
	   else echo "Possible problem with ex17_1, diffs above"; fi; \
	   ${RM} -f ex17_1.tmp

