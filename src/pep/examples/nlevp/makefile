#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

CFLAGS     =
FFLAGS     =
CPPFLAGS   =
FPPFLAGS   =
LOCDIR     = src/pep/examples/nlevp/
EXAMPLESC  = acoustic_wave_1d.c acoustic_wave_2d.c butterfly.c damped_beam.c \
             pdde_stability.c planar_waveguide.c sleeper.c spring.c wiresaw.c loaded_string.c
EXAMPLESF  =
MANSEC     = PEP
TESTS      = acoustic_wave_1d acoustic_wave_2d butterfly damped_beam \
             pdde_stability planar_waveguide sleeper spring wiresaw loaded_string

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

wiresaw: wiresaw.o chkopts
	-${CLINKER} -o wiresaw wiresaw.o ${SLEPC_PEP_LIB}
	${RM} wiresaw.o

loaded_string: loaded_string.o chkopts
	-${CLINKER} -o loaded_string loaded_string.o ${SLEPC_PEP_LIB}
	${RM} loaded_string.o

