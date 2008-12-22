#requirespackage   'PETSC_USING_F90'
#
#   Makes Fortran module functions from *.h90 files in include/finclude
#
ALL: buildmod
speciallib: buildmod
specialfastlib: buildmod
SPECIALLIB     = yes
SPECIALFASTLIB = yes


CFLAGS	   = 
FFLAGS	   = 
SOURCEC	   = 
SOURCEF	   = slepcmod.F
SOURCEH	   = 
OBJSC	   = 
OBJSF	   = slepcmod.o
LIBBASE	   = libslepc
MANSEC	   = Sys
LOCDIR	   = src/sys/f90-mod/
CLEANFILES = *.mod

include ${SLEPC_DIR}/conf/slepc_common

buildmod:
	-@${OMAKE}  clean
	@${OMAKE}  slepcmod.o
	@${OMAKE}  modcopy
	@${OMAKE}  clean
modcopy:
	${CP} -f *.mod ${SLEPC_DIR}/${PETSC_ARCH}/include
