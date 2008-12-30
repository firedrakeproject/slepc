#
# This is the makefile for installing SLEPc. See the Users Manual 
# for directions on installing SLEPc.
#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     SLEPc - Scalable Library for Eigenvalue Problem Computations
#     Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain
#
#     This file is part of SLEPc. See the README file for conditions of use
#     and additional information.
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

ALL: all
LOCDIR = .
DIRS   = src include docs 

include ${SLEPC_DIR}/conf/slepc_common

#
# Basic targets to build SLEPc libraries.
# all: builds the C/C++ and Fortran libraries
all:
	@${OMAKE}  PETSC_ARCH=${PETSC_ARCH} chkpetsc_dir
	@${OMAKE}  PETSC_ARCH=${PETSC_ARCH} chkslepc_dir
	-@${OMAKE} all_build 2>&1 | tee ${PETSC_ARCH}/conf/make.log
	-@egrep -i "( error | error:)" ${PETSC_ARCH}/conf/make.log > /dev/null; if [ "$$?" = "0" ]; then \
           echo "********************************************************************"; \
           echo "  Error during compile, check ${PETSC_ARCH}/conf/make.log"; \
           echo "  Send all contents of ${PETSC_ARCH}/conf to slepc-maint@grycap.upv.es";\
           echo "********************************************************************"; \
           exit 1; \
	 elif [ "${SLEPC_INSTALL_DIR}" == "${SLEPC_DIR}/${PETSC_ARCH}" ]; then \
           echo "Now to check if the libraries are working do: make test";\
           echo "=========================================";\
	 else \
	   echo "Now to install the libraries do: make install";\
	   echo "=========================================";\
	 fi
	
all_build: chk_petsc_dir chk_slepc_dir chklib_dir info deletelibs build shared_nomesg_noinstall
#
# Prints information about the system and version of SLEPc being compiled
#
info:
	-@echo "=========================================="
	-@echo On `date` on `hostname`
	-@echo Machine characteristics: `uname -a`
	-@echo "-----------------------------------------"
	-@echo "Using SLEPc directory: ${SLEPC_DIR}"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "-----------------------------------------"
	-@grep "define SLEPC_VERSION" ${SLEPC_DIR}/include/slepcversion.h | ${SED} "s/........//"
	-@echo "-----------------------------------------"
	-@grep "define PETSC_VERSION" ${PETSC_DIR}/include/petscversion.h | ${SED} "s/........//"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc configure options " ${CONFIGURE_OPTIONS}
	-@echo "Using SLEPc configuration flags:"
	-@cat ${SLEPC_DIR}/${PETSC_ARCH}/conf/slepcvariables
	-@echo "Using PETSc configuration flags:"
	-@if [ -e ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h ]; then \
	   grep "\#define " ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h; \
          else \
	   grep "\#define " ${PETSC_DIR}/include/petscconf.h; \
          fi
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${SLEPC_INCLUDE} ${PETSC_INCLUDE}"
	-@echo "------------------------------------------"
	-@echo "Using C/C++ compiler: ${PCC} ${PCC_FLAGS} ${COPTFLAGS} ${CFLAGS}"
	-@echo "C/C++ Compiler version: " `${CCV}`
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran compiler: ${FC} ${FC_FLAGS} ${FFLAGS} ${FPP_FLAGS}";\
	   echo "Fortran Compiler version: " `${FCV}`;\
         fi
	-@echo "-----------------------------------------"
	-@echo "Using C/C++ linker: ${PCC_LINKER}"
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran linker: ${FC_LINKER}";\
         fi
	-@echo "-----------------------------------------"
	-@echo "Using libraries: ${SLEPC_LIB}"
	-@echo "------------------------------------------"
	-@echo "Using mpirun: ${MPIRUN}"
	-@echo "=========================================="

#
# Builds the SLEPc libraries
#
build:
	-@echo "BEGINNING TO COMPILE SLEPc LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} ACTION=libfast  tree 
	${RANLIB} ${SLEPC_LIB_DIR}/*.${AR_LIB_SUFFIX}
	-@echo "Completed building SLEPc libraries"
	-@echo "========================================="

# Simple test examples for checking a correct installation
test: 
	-@echo "Running test examples to verify correct installation"
	@cd src/examples; ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} testex1
	@if [ "${FC}" != "" ]; then cd src/examples; ${OMAKE} PETSC_ARCH=${PETSC_ARCH} SLEPC_DIR=${SLEPC_DIR} PETSC_DIR=${PETSC_DIR} testex1f; fi;
	-@echo "Completed test examples"

# Builds SLEPc test examples for C
testexamples: info
	-@echo "BEGINNING TO COMPILE AND RUN SLEPc TEST EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_C  tree 
	-@echo "Completed compiling and running test examples"
	-@echo "========================================="

# Builds SLEPc test examples for Fortran
testfortran: info
	-@echo "BEGINNING TO COMPILE AND RUN SLEPc FORTRAN TEST EXAMPLES"
	-@echo "========================================="
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines or the way Fortran formats numbers"
	-@echo "some of the results may not match exactly."
	-@echo "========================================="
	-@if [ "${FC}" != "" ]; then \
	    ${OMAKE} PETSC_ARCH=${PETSC_ARCH} ACTION=testexamples_Fortran  tree ; \
            echo "Completed compiling and running Fortran test examples"; \
          else \
            echo "Error: No FORTRAN compiler available"; \
          fi
	-@
	-@echo "========================================="

# Uni-processor examples in C
testexamples_uni: info
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_C_X11_MPIUni  tree 
	-@echo "Completed compiling and running uniprocessor test examples"
	-@echo "========================================="

# Uni-processor examples in Fortran
testfortran_uni: info
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR FORTRAN EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@if [ "${FC}" != "" ]; then \
            ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=testexamples_Fortran_MPIUni  tree; \
            echo "Completed compiling and running uniprocessor Fortran test examples"; \
          else \
            echo "Error: No FORTRAN compiler available"; \
          fi
	-@
	-@echo "========================================="

# Ranlib on the libraries
ranlib:
	${RANLIB} ${SLEPC_LIB_DIR}/*.${LIB_SUFFIX}

# Deletes SLEPc libraries
deletelibs:
	-${RM} -f ${SLEPC_LIB_DIR}/*

# Cleans up build
allclean: deletelibs
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=clean tree

#
# Check if PETSC_DIR variable specified is valid
#
chk_petsc_dir:
	@if [ ! -f ${PETSC_DIR}/include/petscversion.h ]; then \
	  echo "Incorrect PETSC_DIR specified: ${PETSC_DIR}!"; \
	  echo "You need to use / to separate directories, not \\!"; \
	  echo "Aborting build"; \
	  false; fi
#
# Check if SLEPC_DIR variable specified is valid
#
chk_slepc_dir:
	@if [ ! -f ${SLEPC_DIR}/include/slepcversion.h ]; then \
	  echo "Incorrect SLEPC_DIR specified: ${SLEPC_DIR}!"; \
	  echo "You need to use / to separate directories, not \\!"; \
	  echo "Aborting build"; \
	  false; fi

install:
	-@if [ "${PETSC_ARCH}" == "" ]; then \
	  echo "PETSC_ARCH is undefined";\
	elif [ "${SLEPC_INSTALL_DIR}" = "${SLEPC_DIR}/${PETSC_ARCH}" ]; then \
	  echo "Install directory is current directory; nothing needs to be done";\
        else \
	  echo Installing SLEPc at ${SLEPC_INSTALL_DIR};\
          if [ ! -d `dirname ${SLEPC_INSTALL_DIR}` ]; then \
	    ${MKDIR} `dirname ${SLEPC_INSTALL_DIR}` ; \
          fi;\
          if [ ! -d ${SLEPC_INSTALL_DIR} ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR} ; \
          fi;\
          if [ ! -d ${SLEPC_INSTALL_DIR}/include ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR}/include ; \
          fi;\
          cp -f include/*.h ${SLEPC_INSTALL_DIR}/include;\
          if [ -f ${PETSC_ARCH}/include/slepceps.mod ]; then \
            cp -f ${PETSC_ARCH}/include/*.mod ${SLEPC_INSTALL_DIR}/include;\
          fi;\
          if [ ! -d ${SLEPC_INSTALL_DIR}/include/finclude ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR}/include/finclude ; \
          fi;\
          cp -f include/finclude/*.h* ${SLEPC_INSTALL_DIR}/include/finclude;\
          if [ ! -d ${SLEPC_INSTALL_DIR}/include/finclude/ftn-auto ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR}/include/finclude/ftn-auto ; \
          fi;\
          cp -f include/finclude/ftn-auto/*.h90 ${SLEPC_INSTALL_DIR}/include/finclude/ftn-auto;\
          if [ ! -d ${SLEPC_INSTALL_DIR}/include/finclude/ftn-custom ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR}/include/finclude/ftn-custom ; \
          fi;\
          cp -f include/finclude/ftn-custom/*.h90 ${SLEPC_INSTALL_DIR}/include/finclude/ftn-custom;\
          if [ ! -d ${SLEPC_INSTALL_DIR}/include/private ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR}/include/private ; \
          fi;\
          cp -f include/finclude/*.h ${SLEPC_INSTALL_DIR}/include/private;\
          if [ ! -d ${SLEPC_INSTALL_DIR}/conf ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR}/conf ; \
          fi;\
          cp -f conf/slepc_common* ${SLEPC_INSTALL_DIR}/conf;\
          cp -f conf/slepc_common_variables_install ${SLEPC_INSTALL_DIR}/conf/slepc_common_variables;\
          cp -f ${PETSC_ARCH}/conf/slepcvariables ${SLEPC_INSTALL_DIR}/conf;\
          if [ ! -d ${SLEPC_INSTALL_DIR}/lib ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR}/lib ; \
          fi;\
          if [ -d ${PETSC_ARCH}/lib ]; then \
            cp -f ${PETSC_ARCH}/lib/* ${SLEPC_INSTALL_DIR}/lib;\
            ${RANLIB} ${SLEPC_INSTALL_DIR}/lib/*.a ;\
            ${OMAKE} PETSC_ARCH="" SLEPC_DIR=${SLEPC_INSTALL_DIR} shared; \
          fi;\
	  echo "If using sh/bash, do the following:";\
          echo "  SLEPC_DIR="${SLEPC_INSTALL_DIR}"; export SLEPC_DIR";\
          echo "  unset PETSC_ARCH";\
          echo "If using csh/tcsh, do the following:";\
          echo "  setenv SLEPC_DIR "${SLEPC_INSTALL_DIR};\
          echo "  unsetenv PETSC_ARCH";\
          echo "Run the following to verify the install (remain in current directory for the tests):";\
          echo "  make test";\
        fi;

# ------------------------------------------------------------------
#
# All remaining actions are intended for SLEPc developers only.
# SLEPc users should not generally need to use these commands.
#

# Builds all the documentation
alldoc: alldoc1 alldoc2

# Build everything that goes into 'doc' dir except html sources
alldoc1: chk_loc deletemanualpages
	-${OMAKE} ACTION=manualpages_buildcite tree_basic LOC=${LOC}
	-@sed -e s%man+../%man+manualpages/% ${LOC}/docs/manualpages/manualpages.cit > ${LOC}/docs/manualpages/htmlmap
	-@cat ${PETSC_DIR}/src/docs/mpi.www.index >> ${LOC}/docs/manualpages/htmlmap
	-${OMAKE} ACTION=slepc_manualpages tree_basic LOC=${LOC}
	-${PETSC_DIR}/bin/maint/wwwindex.py ${SLEPC_DIR} ${LOC}
	-${OMAKE} ACTION=manexamples tree_basic LOC=${LOC}

# Builds .html versions of the source
# html overwrites some stuff created by update-docs - hence this is done later.
alldoc2: chk_loc
	-${OMAKE} ACTION=slepc_html PETSC_DIR=${PETSC_DIR} alltree LOC=${LOC}
	cp ${LOC}/docs/manual.htm ${LOC}/docs/index.html

# Deletes man pages (HTML version)
deletemanualpages: chk_loc
	-@if [ -d ${LOC} -a -d ${LOC}/docs/manualpages ]; then \
          find ${LOC}/docs/manualpages -type f -name "*.html" -exec ${RM} {} \; ;\
          ${RM} ${LOC}/docs/manualpages/manualpages.cit ;\
        fi

# Builds Fortran stub files
allfortranstubs:
	-@${SLEPC_DIR}/config/generatefortranstubs.py ${BFORT}

# -------------------------------------------------------------------------------
#
# Some macros to check if the Fortran interface is up-to-date.
#
countfortranfunctions: 
	-@for D in `find ${SLEPC_DIR}/src -name ftn-auto` \
	`find ${SLEPC_DIR}/src -name ftn-custom`; do cd $$D; \
	egrep '^void' *.c | \
	cut -d'(' -f1 | tr -s  ' ' | cut -d' ' -f3 | uniq | egrep -v "(^$$|Petsc)" | \
	sed "s/_$$//"; done | sort > /tmp/countfortranfunctions

countcfunctions:
	-@ grep "EXTERN " ${SLEPC_DIR}/include/*.h | grep "(" | tr -s ' ' | \
	cut -d'(' -f1 | cut -d' ' -f3 | grep -v "\*" | tr -s '\012' |  \
	tr 'A-Z' 'a-z' |  sort > /tmp/countcfunctions

difffortranfunctions: countfortranfunctions countcfunctions
	-@echo -------------- Functions missing in the Fortran interface ---------------------
	-@${DIFF} /tmp/countcfunctions /tmp/countfortranfunctions | grep "^<" | cut -d' ' -f2
	-@echo ----------------- Functions missing in the C interface ------------------------
	-@${DIFF} /tmp/countcfunctions /tmp/countfortranfunctions | grep "^>" | cut -d' ' -f2
	-@${RM}  /tmp/countcfunctions /tmp/countfortranfunctions

checkbadfortranstubs:
	-@echo "========================================="
	-@echo "Functions with MPI_Comm as an Argument"
	-@echo "========================================="
	-@for D in `find ${SLEPC_DIR}/src -name ftn-auto`; do cd $$D; \
	grep '^void' *.c | grep 'MPI_Comm' | \
	tr -s ' ' | tr -s ':' ' ' |cut -d'(' -f1 | cut -d' ' -f1,3; done
	-@echo "========================================="
	-@echo "Functions with a String as an Argument"
	-@echo "========================================="
	-@for D in `find ${SLEPC_DIR}/src -name ftn-auto`; do cd $$D; \
	grep '^void' *.c | grep 'char \*' | \
	tr -s ' ' | tr -s ':' ' ' |cut -d'(' -f1 | cut -d' ' -f1,3; done
	-@echo "========================================="
	-@echo "Functions with Pointers to PETSc Objects as Argument"
	-@echo "========================================="
	-@_p_OBJ=`grep _p_ ${PETSC_DIR}/include/*.h | tr -s ' ' | \
	cut -d' ' -f 3 | tr -s '\012' | grep -v '{' | cut -d'*' -f1 | \
	sed "s/_p_//g" | tr -s '\012 ' ' *|' ` ; \
	_p_OBJS=`grep _p_ ${SLEPC_DIR}/include/*.h | tr -s ' ' | \
	cut -d' ' -f 3 | tr -s '\012' | grep -v '{' | cut -d'*' -f1 | \
	sed "s/_p_//g" | tr -s '\012 ' ' *|' ` ; \
	for D in `find ${SLEPC_DIR}/src -name ftn-auto`; do cd $$D; \
	for OBJ in $$_p_OBJ $$_p_OBJS; do \
	grep "$$OBJ \*" *.c | tr -s ' ' | tr -s ':' ' ' | \
	cut -d'(' -f1 | cut -d' ' -f1,4; \
	done; done

# Generate tags with Exuberant Ctags 5.5.4
ctags:
	-@ctags -R --exclude="examples" --languages=-HTML,Make --fortran-kinds=-l include src
	-@egrep -v __FUNCT__\|PetscToPointer\|PetscFromPointer\|PetscRmPointer tags > tags-tmp
	-@mv tags-tmp tags

