#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
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
             pdde_stability.c planar_waveguide.c sleeper.c spring.c wiresaw.c
EXAMPLESF  =
MANSEC     = PEP
TESTS      = acoustic_wave_1d acoustic_wave_2d butterfly damped_beam \
             pdde_stability planar_waveguide sleeper spring wiresaw

TESTEXAMPLES_C                     = sleeper.PETSc runsleeper_1 sleeper.rm
TESTEXAMPLES_C_NOTSINGLE           = damped_beam.PETSc rundamped_beam_1 damped_beam.rm \
                                     sleeper.PETSc runsleeper_2_toar runsleeper_3 sleeper.rm \
                                     spring.PETSc runspring_1 runspring_3 runspring_4 spring.rm
TESTEXAMPLES_C_NOCOMPLEX           = butterfly.PETSc runbutterfly_1 butterfly.rm
TESTEXAMPLES_C_DATAFILE            = butterfly.PETSc runbutterfly_2 butterfly.rm
TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE = acoustic_wave_1d.PETSc runacoustic_wave_1d_1 runacoustic_wave_1d_2 \
                                                            runacoustic_wave_1d_3 acoustic_wave_1d.rm \
                                     acoustic_wave_2d.PETSc runacoustic_wave_2d_1 acoustic_wave_2d.rm \
                                     planar_waveguide.PETSc runplanar_waveguide_1 planar_waveguide.rm \
                                     wiresaw.PETSc runwiresaw_1 wiresaw.rm
TESTEXAMPLES_C_COMPLEX             = damped_beam.PETSc rundamped_beam_1_jd damped_beam.rm \
                                     pdde_stability.PETSc runpdde_stability_1 pdde_stability.rm \
                                     sleeper.PETSc runsleeper_2_jd sleeper.rm \
                                     spring.PETSc runspring_2_jd spring.rm

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

#------------------------------------------------------------------------------------
runacoustic_wave_1d_1: runacoustic_wave_1d_1_toar runacoustic_wave_1d_1_stoar runacoustic_wave_1d_1_qarnoldi runacoustic_wave_1d_1_linear
runacoustic_wave_1d_1_%:
	-@${SETTEST}; check=acoustic_wave_1d_1; pep=$*; \
	if [ "$$pep" = stoar ]; then pep="stoar -pep_hermitian -pep_stoar_locking 0"; fi; \
	${MPIEXEC} -n 1 ./acoustic_wave_1d -pep_type $${pep} -st_transform -pep_nev 4 -pep_tol 1e-7 -n 24 -st_type sinvert -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runacoustic_wave_1d_2: runacoustic_wave_1d_2_none runacoustic_wave_1d_2_norm runacoustic_wave_1d_2_residual
runacoustic_wave_1d_2_%:
	-@${SETTEST}; check=acoustic_wave_1d_1; extract=$*; \
	if [ "$$pep" = stoar ]; then pep="stoar -pep_hermitian"; fi; \
	${MPIEXEC} -n 1 ./acoustic_wave_1d -pep_extract $${extract} -st_transform -pep_nev 4 -pep_tol 1e-7 -n 24 -st_type sinvert -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runacoustic_wave_1d_3: runacoustic_wave_1d_3_none runacoustic_wave_1d_3_norm runacoustic_wave_1d_3_residual
runacoustic_wave_1d_3_%:
	-@${SETTEST}; check=acoustic_wave_1d_1; extract=$*; \
	if [ "$$pep" = stoar ]; then pep="stoar -pep_hermitian"; fi; \
	${MPIEXEC} -n 1 ./acoustic_wave_1d -pep_type linear -pep_extract $${extract} -st_transform -pep_nev 4 -pep_tol 1e-7 -n 24 -st_type sinvert -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runacoustic_wave_2d_1: runacoustic_wave_2d_1_toar runacoustic_wave_2d_1_stoar runacoustic_wave_2d_1_qarnoldi runacoustic_wave_2d_1_linear
runacoustic_wave_2d_1_%:
	-@${SETTEST}; check=acoustic_wave_2d_1; pep=$*; \
	if [ "$$pep" = stoar ]; then pep="stoar -pep_hermitian"; \
	elif [ "$$pep" = toar ]; then pep="toar -pep_toar_locking 0"; fi; \
	${MPIEXEC} -n 1 ./acoustic_wave_2d -pep_type $${pep} -pep_nev 2 -pep_ncv 18 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runbutterfly_1: runbutterfly_1_toar runbutterfly_1_linear
runbutterfly_1_%:
	-@${SETTEST}; check=butterfly_1; pep=$*; \
	if [ "$$pep" = toar ]; then pep="toar -pep_toar_restart 0.3"; fi; \
	${MPIEXEC} -n 1 ./butterfly -pep_type $${pep} -pep_nev 4 -st_type sinvert -pep_target 0.01 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runbutterfly_2: runbutterfly_2_toar runbutterfly_2_linear
runbutterfly_2_%:
	-@${SETTEST}; check=butterfly_2; pep=$*; \
	${MPIEXEC} -n 1 ./butterfly -pep_type $${pep} -pep_nev 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

rundamped_beam_1: rundamped_beam_1_toar rundamped_beam_1_qarnoldi rundamped_beam_1_linear
rundamped_beam_1_%:
	-@${SETTEST}; check=damped_beam_1; pep=$*; \
	if [ "${PETSC_SCALAR}" = "complex" ]; then target="1i"; else target="0"; fi; \
	if [ "$$pep" != jd ]; then sinv="-st_type sinvert"; fi; \
	if [ "$$pep" = qarnoldi ]; then pep="qarnoldi -pep_qarnoldi_locking 0"; fi; \
	${MPIEXEC} -n 1 ./damped_beam -pep_type $${pep} -pep_nev 2 -pep_ncv 12 -pep_target $$target $$sinv -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runpdde_stability_1: runpdde_stability_1_toar runpdde_stability_1_qarnoldi runpdde_stability_1_linear
runpdde_stability_1_%:
	-@${SETTEST}; check=pdde_stability_1; pep=$*; \
	${MPIEXEC} -n 1 ./pdde_stability -pep_type $${pep} -pep_ncv 25 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runplanar_waveguide_1: runplanar_waveguide_1_toar runplanar_waveguide_1_linear
runplanar_waveguide_1_%:
	-@${SETTEST}; check=planar_waveguide_1; pep=$*; \
	if [ "$$pep" = jd ]; then sinv="-pep_target -.1"; else sinv="-st_type sinvert"; fi; \
	${MPIEXEC} -n 1 ./planar_waveguide -pep_type $${pep} -pep_nev 4 $$sinv -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runsleeper_1: runsleeper_1_toar runsleeper_1_qarnoldi runsleeper_1_linear
runsleeper_1_%:
	-@${SETTEST}; check=sleeper_1; pep=$*; \
	if [ "$$pep" = qarnoldi ]; then pep="qarnoldi -pep_qarnoldi_restart 0.4"; fi; \
	${MPIEXEC} -n 1 ./sleeper -n 100 -pep_type $${pep} -pep_nev 4 -pep_ncv 24 -st_type sinvert -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runsleeper_2: runsleeper_2_toar runsleeper_2_jd
runsleeper_2_%:
	-@${SETTEST}; check=sleeper_2; pep=$*; \
	if [ "$$pep" = toar ]; then pep="toar -pep_toar_restart .35 -pep_ncv 10 -pep_max_it 2000 -st_type sinvert"; \
	elif [ "$$pep" = jd ]; then pep="jd -pep_target -.61"; fi; \
	${MPIEXEC} -n 1 ./sleeper -n 100 -pep_type $${pep} -pep_nev 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runsleeper_3:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./sleeper -n 270 -pep_type stoar -pep_hermitian -st_type sinvert -pep_nev 2 -pep_target -.89 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runspring_1: runspring_1_toar runspring_1_stoar runspring_1_qarnoldi runspring_1_linear
runspring_1_%:
	-@${SETTEST}; check=spring_1; pep=$*; \
	if [ "$$pep" != jd ]; then sinv="-st_type sinvert"; fi; \
	if [ "$$pep" = stoar ]; then pep="stoar -pep_hermitian -pep_conv_rel"; \
	elif [ "$$pep" = qarnoldi ]; then pep="qarnoldi -pep_conv_rel"; \
	else pep="$$pep -pep_conv_norm "; fi; \
	${MPIEXEC} -n 1 ./spring -pep_type $${pep} -pep_nev 4 -n 24 -pep_ncv 18 -pep_target -.5 $$sinv -pep_scale diagonal -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runspring_2_jd:
	-@${SETTEST}; check=spring_2; \
	${MPIEXEC} -n 1 ./spring -pep_type jd -pep_nev 4 -n 24 -pep_ncv 18 -pep_target -50 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runspring_3:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./spring -n 300 -pep_hermitian -pep_interval -10.1,-9.5 -pep_type stoar -st_type sinvert -st_pc_type cholesky -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runspring_4:
	-@${SETTEST}; \
	${MPIEXEC} -n 1 ./spring -n 300 -pep_hyperbolic -pep_interval -9.6,-.527 -pep_type stoar -st_type sinvert -st_pc_type cholesky -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

runwiresaw_1: runwiresaw_1_toar runwiresaw_1_qarnoldi runwiresaw_1_linear_h1 runwiresaw_1_linear_h2
runwiresaw_1_%:
	-@${SETTEST}; check=wiresaw_1; pep=$*; \
	if [ "$$pep" = linear_h1 ]; then pep="linear -pep_linear_explicitmatrix -pep_linear_cform 1 -pep_linear_st_ksp_type bcgs -pep_linear_st_pc_type kaczmarz"; \
	elif [ "$$pep" = linear_h2 ]; then pep="linear -pep_linear_explicitmatrix -pep_linear_cform 2"; fi; \
	${MPIEXEC} -n 1 ./wiresaw -pep_type $${pep} -pep_nev 4 -terse > $${test}.tmp 2>&1; \
	${TESTCODE}

