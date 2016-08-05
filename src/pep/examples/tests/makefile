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
LOCDIR     = src/pep/examples/tests/
EXAMPLESC  = test1.c test2.c test3.c
EXAMPLESF  = test3f.F
MANSEC     = PEP
TESTS      = test1 test2 test3 test3f

TESTEXAMPLES_C                     = test2.PETSc runtest2_2 runtest2_4 test2.rm \
                                     test3.PETSc runtest3_1 test3.rm
TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE = test1.PETSc runtest1_1 test1.rm
TESTEXAMPLES_C_NOTSINGLE           = test2.PETSc runtest2_1 runtest2_3 test2.rm
TESTEXAMPLES_FORTRAN               = test3f.PETSc runtest3f_1 test3f.rm \

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

test1: test1.o chkopts
	-${CLINKER} -o test1 test1.o ${SLEPC_PEP_LIB}
	${RM} test1.o

test2: test2.o chkopts
	-${CLINKER} -o test2 test2.o ${SLEPC_PEP_LIB}
	${RM} test2.o

test3: test3.o chkopts
	-${CLINKER} -o test3 test3.o ${SLEPC_PEP_LIB}
	${RM} test3.o

test3f: test3f.o chkopts
	-${FLINKER} -o test3f test3f.o ${SLEPC_PEP_LIB}
	${RM} test3f.o

#------------------------------------------------------------------------------------

runtest1_1: runtest1_1_toar runtest1_1_qarnoldi runtest1_1_linear runtest1_1_linear_gd
runtest1_1_%:
	-@test=$(subst run,,$@); check=test1_1; pep=$*; \
	if [ "$$pep" = linear_gd ]; then pep="linear -epstype gd"; fi; \
	${MPIEXEC} -n 1 ./test1 -type $${pep} -m 11 >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_1: runtest2_1_toar runtest2_1_qarnoldi runtest2_1_linear runtest2_1_linear_gd
runtest2_1_%:
	-@test=$(subst run,,$@); check=test2_1; pep=$*; \
	if [ "$$pep" = linear_gd ]; then pep="linear -pep_linear_eps_type gd -pep_linear_explicitmatrix"; \
	elif [ "$$pep" = qarnoldi ]; then pep="qarnoldi -st_transform"; fi; \
	${MPIEXEC} -n 1 ./test2 -pep_type $${pep} -pep_nev 4 >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_2: runtest2_2_toar runtest2_2_toar_scaleboth runtest2_2_toar_transform runtest2_2_qarnoldi runtest2_2_linear
runtest2_2_%:
	-@test=$(subst run,,$@); check=test2_2; pep=$*; \
	if [ "$$pep" = toar_scaleboth ]; then pep="toar -pep_scale both"; \
	elif [ "$$pep" = toar_transform ]; then pep="toar -st_transform"; \
	elif [ "$$pep" = qarnoldi ]; then pep="qarnoldi -st_transform"; fi; \
	${MPIEXEC} -n 1 ./test2 -pep_type $${pep} -st_type sinvert -pep_target -0.4 -pep_nev 4 >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_3: runtest2_3_none runtest2_3_norm runtest2_3_residual runtest2_3_structured
runtest2_3_%:
	-@test=$(subst run,,$@); check=test2_3; extract=$*; \
	${MPIEXEC} -n 1 ./test2 -pep_nev 12 -pep_extract $${extract} >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest2_4: runtest2_4_schur runtest2_4_mbe runtest2_4_explicit runtest2_4_multiple
runtest2_4_%:
	-@test=$(subst run,,$@); check=test2_4; refine=$*; \
	if [ "$$refine" = schur ]; then refine="simple -pep_refine_scheme schur"; \
	elif [ "$$refine" = mbe ]; then refine="simple -pep_refine_scheme mbe -pep_refine_ksp_type preonly -pep_refine_pc_type lu"; \
	elif [ "$$refine" = explicit ]; then refine="simple -pep_refine_scheme explicit"; fi; \
	${MPIEXEC} -n 1 ./test2 -st_type sinvert -pep_target -0.4 -pep_nev 4 -pep_refine $${refine} >> $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3_1:
	-@test=$(subst run,,$@); check=$$test; \
	${MPIEXEC} -n 1 ./test3 -pep_tol 1e-6 -pep_ncv 22 > $${test}.tmp 2>&1; \
	${TESTCODE}

runtest3f_1:
	-@test=$(subst run,,$@); check=$$test; \
	${MPIEXEC} -n 1 ./test3f -pep_tol 1e-6 -pep_ncv 22 > $${test}.tmp 2>&1; \
	${TESTCODE}

