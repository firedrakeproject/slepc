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
LOCDIR     = src/pep/examples/nlevp/
EXAMPLESC  = acoustic_wave_1d.c acoustic_wave_2d.c butterfly.c damped_beam.c \
             pdde_stability.c planar_waveguide.c sleeper.c spring.c
EXAMPLESF  = 
MANSEC     = PEP
TESTS      = acoustic_wave_1d acoustic_wave_2d butterfly damped_beam \
             pdde_stability planar_waveguide sleeper spring

TESTEXAMPLES_C                     = sleeper.PETSc runsleeper_1 sleeper.rm \
                                     spring.PETSc runspring_1 spring.rm
TESTEXAMPLES_C_NOCOMPLEX           = butterfly.PETSc runbutterfly_1 butterfly.rm
TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE = acoustic_wave_1d.PETSc runacoustic_wave_1d_1 acoustic_wave_1d.rm \
                                     acoustic_wave_2d.PETSc runacoustic_wave_2d_1 acoustic_wave_2d.rm \
                                     damped_beam.PETSc rundamped_beam_1 damped_beam.rm \
                                     planar_waveguide.PETSc runplanar_waveguide_1 planar_waveguide.rm
TESTEXAMPLES_C_COMPLEX             = pdde_stability.PETSc runpdde_stability_1 pdde_stability.rm

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

acoustic_wave_1d: acoustic_wave_1d.o chkopts
	-${CLINKER} -o acoustic_wave_1d acoustic_wave_1d.o ${SLEPC_PEP_LIB}
	${RM} acoustic_wave_1d.o

acoustic_wave_2d: acoustic_wave_2d.o chkopts
	-${CLINKER} -o acoustic_wave_2d acoustic_wave_2d.o ${SLEPC_PEP_LIB}
	${RM} acoustic_wave_2d.o

butterfly: butterfly.o chkopts
	-${CLINKER} -o butterfly butterfly.o ${SLEPC_PEP_LIB}
	${RM} butterfly.o

damped_beam: damped_beam.o chkopts
	-${CLINKER} -o damped_beam damped_beam.o ${SLEPC_PEP_LIB}
	${RM} damped_beam.o

pdde_stability: pdde_stability.o chkopts
	-${CLINKER} -o pdde_stability pdde_stability.o ${SLEPC_PEP_LIB}
	${RM} pdde_stability.o

planar_waveguide: planar_waveguide.o chkopts
	-${CLINKER} -o planar_waveguide planar_waveguide.o ${SLEPC_PEP_LIB}
	${RM} planar_waveguide.o

sleeper: sleeper.o chkopts
	-${CLINKER} -o sleeper sleeper.o ${SLEPC_PEP_LIB}
	${RM} sleeper.o

spring: spring.o chkopts
	-${CLINKER} -o spring spring.o ${SLEPC_PEP_LIB}
	${RM} spring.o

#------------------------------------------------------------------------------------
runacoustic_wave_1d_1:
	-@test=$(subst run,,$@); check=$$test; \
	${MPIEXEC} -n 1 ./acoustic_wave_1d -pep_nev 4 -pep_tol 1e-7 -n 24 -st_type sinvert -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runacoustic_wave_2d_1:
	-@test=$(subst run,,$@); check=$$test; \
	${MPIEXEC} -n 1 ./acoustic_wave_2d -pep_nev 2 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runbutterfly_1:
	-@test=$(subst run,,$@); check=$$test; \
	${MPIEXEC} -n 1 ./butterfly -pep_nev 4 -st_type sinvert -pep_target 0.01 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

rundamped_beam_1:
	-@test=$(subst run,,$@); check=$$test; \
	${MPIEXEC} -n 1 ./damped_beam -pep_nev 2 -pep_ncv 12 -st_type sinvert -pep_target -7 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runpdde_stability_1:
	-@test=$(subst run,,$@); check=$$test; \
	${MPIEXEC} -n 1 ./pdde_stability -pep_ncv 25 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runplanar_waveguide_1:
	-@test=$(subst run,,$@); check=$$test; \
	${MPIEXEC} -n 1 ./planar_waveguide -pep_nev 4 -st_type sinvert -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runsleeper_1:
	-@test=$(subst run,,$@); check=$$test; \
	${MPIEXEC} -n 1 ./sleeper -pep_nev 4 -pep_ncv 24 -st_type sinvert -n 100 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runspring_1:
	-@test=$(subst run,,$@); check=$$test; \
	${MPIEXEC} -n 1 ./spring -pep_nev 4 -n 24 -st_type sinvert -pep_ncv 18 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

