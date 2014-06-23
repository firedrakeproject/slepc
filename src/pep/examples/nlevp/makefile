#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain
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
EXAMPLESC  = acoustic_wave_1d.c acoustic_wave_2d.c butterfly.c sleeper.c spring.c
EXAMPLESF  = 
MANSEC     = PEP
TESTS      = acoustic_wave_1d acoustic_wave_2d butterfly sleeper spring

TESTEXAMPLES_C = acoustic_wave_1d.PETSc runacoustic_wave_1d_1 acoustic_wave_1d.rm \
                 acoustic_wave_2d.PETSc runacoustic_wave_2d_1 acoustic_wave_2d.rm \
                 butterfly.PETSc runbutterfly butterfly.rm \
                 sleeper.PETSc runsleeper sleeper.rm \
                 spring.PETSc runspring spring.rm
TESTEXAMPLES_C_NOCOMPLEX = 

include ${SLEPC_DIR}/conf/slepc_common

acoustic_wave_1d: acoustic_wave_1d.o chkopts
	-${CLINKER} -o acoustic_wave_1d acoustic_wave_1d.o ${SLEPC_PEP_LIB}
	${RM} acoustic_wave_1d.o

acoustic_wave_2d: acoustic_wave_2d.o chkopts
	-${CLINKER} -o acoustic_wave_2d acoustic_wave_2d.o ${SLEPC_PEP_LIB}
	${RM} acoustic_wave_2d.o

butterfly: butterfly.o chkopts
	-${CLINKER} -o butterfly butterfly.o ${SLEPC_PEP_LIB}
	${RM} butterfly.o

sleeper: sleeper.o chkopts
	-${CLINKER} -o sleeper sleeper.o ${SLEPC_PEP_LIB}
	${RM} sleeper.o

spring: spring.o chkopts
	-${CLINKER} -o spring spring.o ${SLEPC_PEP_LIB}
	${RM} spring.o

#------------------------------------------------------------------------------------
DATAPATH = ${SLEPC_DIR}/share/slepc/datafiles/matrices

runacoustic_wave_1d_1:
	-@${MPIEXEC} -np 1 ./acoustic_wave_1d -pep_nev 4 -n 20 -pep_terse > acoustic_wave_1d_1.tmp 2>&1; \
	   if (${DIFF} output/acoustic_wave_1d_1.out acoustic_wave_1d_1.tmp) then true; \
	   else echo "Possible problem with acoustic_wave_1d_1, diffs above"; fi; \
	   ${RM} -f acoustic_wave_1d_1.tmp

runacoustic_wave_2d_1:
	-@${MPIEXEC} -np 1 ./acoustic_wave_2d -pep_nev 4 -n 20 -pep_terse > acoustic_wave_2d_1.tmp 2>&1; \
	   if (${DIFF} output/acoustic_wave_2d_1.out acoustic_wave_2d_1.tmp) then true; \
	   else echo "Possible problem with acoustic_wave_2d_1, diffs above"; fi; \
	   ${RM} -f acoustic_wave_2d_1.tmp

runsleeper:
	-@${MPIEXEC} -np 1 ./sleeper -pep_nev 4 -n 20 -pep_terse > sleeper_1.tmp 2>&1; \
	   if (${DIFF} output/sleeper_1.out sleeper_1.tmp) then true; \
	   else echo "Possible problem with sleeper_1, diffs above"; fi; \
	   ${RM} -f sleeper_1.tmp

runbutterfly:
	-@${MPIEXEC} -np 1 ./butterfly -pep_nev 4 -n 20 -pep_terse > butterfly_1.tmp 2>&1; \
	   if (${DIFF} output/butterfly_1.out butterfly_1.tmp) then true; \
	   else echo "Possible problem with butterfly_1, diffs above"; fi; \
	   ${RM} -f butterfly_1.tmp

runspring:
	-@${MPIEXEC} -np 1 ./spring -pep_nev 4 -n 20 -pep_terse > spring_1.tmp 2>&1; \
	   if (${DIFF} output/spring_1.out spring_1.tmp) then true; \
	   else echo "Possible problem with spring_1, diffs above"; fi; \
	   ${RM} -f spring_1.tmp

