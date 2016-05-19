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

TESTEXAMPLES_C = acoustic_wave_1d.PETSc runacoustic_wave_1d_1 acoustic_wave_1d.rm \
                 acoustic_wave_2d.PETSc runacoustic_wave_2d_1 acoustic_wave_2d.rm \
                 butterfly.PETSc runbutterfly butterfly.rm \
                 damped_beam.PETSc rundamped_beam damped_beam.rm \
                 planar_waveguide.PETSc runplanar_waveguide planar_waveguide.rm \
                 sleeper.PETSc runsleeper sleeper.rm \
                 spring.PETSc runspring spring.rm
TESTEXAMPLES_C_COMPLEX = pdde_stability.PETSc runpdde_stability pdde_stability.rm

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
	-@${MPIEXEC} -n 1 ./acoustic_wave_1d -pep_nev 4 -pep_tol 1e-7 -n 24 -st_type sinvert -terse > acoustic_wave_1d_1.tmp 2>&1; \
	   if (${DIFF} output/acoustic_wave_1d_1.out acoustic_wave_1d_1.tmp) then true; \
	   else echo "Possible problem with acoustic_wave_1d_1, diffs above"; fi; \
	   ${RM} -f acoustic_wave_1d_1.tmp

runacoustic_wave_2d_1:
	-@${MPIEXEC} -n 1 ./acoustic_wave_2d -pep_nev 2 -terse > acoustic_wave_2d_1.tmp 2>&1; \
	   if (${DIFF} output/acoustic_wave_2d_1.out acoustic_wave_2d_1.tmp) then true; \
	   else echo "Possible problem with acoustic_wave_2d_1, diffs above"; fi; \
	   ${RM} -f acoustic_wave_2d_1.tmp

runbutterfly:
	-@${MPIEXEC} -n 1 ./butterfly -pep_nev 4 -st_type sinvert -pep_target 0.01 -terse > butterfly_1.tmp 2>&1; \
	   if (${DIFF} output/butterfly_1.out butterfly_1.tmp) then true; \
	   else echo "Possible problem with butterfly_1, diffs above"; fi; \
	   ${RM} -f butterfly_1.tmp

rundamped_beam:
	-@${MPIEXEC} -n 1 ./damped_beam -pep_nev 2 -pep_ncv 12 -st_type sinvert -pep_target -7 -terse > damped_beam_1.tmp 2>&1; \
	   if (${DIFF} output/damped_beam_1.out damped_beam_1.tmp) then true; \
	   else echo "Possible problem with damped_beam_1, diffs above"; fi; \
	   ${RM} -f damped_beam_1.tmp

runpdde_stability:
	-@${MPIEXEC} -n 1 ./pdde_stability -terse > pdde_stability_1.tmp 2>&1; \
	   if (${DIFF} output/pdde_stability_1.out pdde_stability_1.tmp) then true; \
	   else echo "Possible problem with pdde_stability_1, diffs above"; fi; \
	   ${RM} -f pdde_stability_1.tmp

runplanar_waveguide:
	-@${MPIEXEC} -n 1 ./planar_waveguide -pep_nev 4 -st_type sinvert -terse > planar_waveguide_1.tmp 2>&1; \
	   if (${DIFF} output/planar_waveguide_1.out planar_waveguide_1.tmp) then true; \
	   else echo "Possible problem with planar_waveguide_1, diffs above"; fi; \
	   ${RM} -f planar_waveguide_1.tmp

runsleeper:
	-@${MPIEXEC} -n 1 ./sleeper -pep_nev 4 -pep_ncv 24 -st_type sinvert -n 100 -terse > sleeper_1.tmp 2>&1; \
	   if (${DIFF} output/sleeper_1.out sleeper_1.tmp) then true; \
	   else echo "Possible problem with sleeper_1, diffs above"; fi; \
	   ${RM} -f sleeper_1.tmp

runspring:
	-@${MPIEXEC} -n 1 ./spring -pep_nev 4 -n 24 -st_type sinvert -pep_ncv 18 -terse > spring_1.tmp 2>&1; \
	   if (${DIFF} output/spring_1.out spring_1.tmp) then true; \
	   else echo "Possible problem with spring_1, diffs above"; fi; \
	   ${RM} -f spring_1.tmp

