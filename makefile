#
# This is the makefile for installing SLEPc. See the Users Manual 
# for directions on installing SLEPc.
#
ALL: all
LOCDIR = .
DIRS   = src include docs 

include ${SLEPC_DIR}/bmake/slepc_common

#
# Basic targets to build SLEPc libraries.
# all: builds the C/C++ and Fortran libraries
all:
	@${OMAKE}  PETSC_ARCH=${PETSC_ARCH} chkpetsc_dir
	@${OMAKE}  PETSC_ARCH=${PETSC_ARCH} chkslepc_dir
	-@${OMAKE} all_build 2>&1 | tee make_log_${PETSC_ARCH}
	-@egrep -i "( error | error:)" make_log_${PETSC_ARCH} > /dev/null; if [ "$$?" = "0" ]; then \
           echo "********************************************************************"; \
           echo "  Error during compile, check make_log_${PETSC_ARCH}"; \
           echo "  Send it and configure.log to slepc-maint@grycap.upv.es";\
           echo "********************************************************************"; \
           exit 1; fi
	
all_build: chk_petsc_dir chk_slepc_dir chklib_dir info deletelibs build shared
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
	-@cat ${SLEPC_DIR}/bmake/${PETSC_ARCH}/slepcconf
	-@echo "Using PETSc configuration flags:"
	-@grep "\#define " ${PETSC_DIR}/bmake/${PETSC_ARCH}/petscconf.h
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
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=libfast  tree 
	${RANLIB} ${SLEPC_LIB_DIR}/*.${AR_LIB_SUFFIX}
	-@echo "Completed building SLEPc libraries"
	-@echo "========================================="

# Builds SLEPc test examples for a given architecture
test: 
	-@echo "Running test examples to verify correct installation"
	@cd src/examples; ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} ex1.PETSc runex1_1
	@if [ "${FC}" != "" ]; then cd src/examples; ${OMAKE} PETSC_ARCH=${PETSC_ARCH} SLEPC_DIR=${SLEPC_DIR} PETSC_DIR=${PETSC_DIR} ex1f.PETSc runex1f_1; fi;
	-@echo "Completed test examples"

testexamples: info
	-@echo "BEGINNING TO COMPILE AND RUN SLEPc TEST EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_C  tree 
	-@echo "Completed compiling and running test examples"
	-@echo "========================================="

# Builds SLEPc test examples for a given architecture
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

# Builds SLEPc test examples for a given architecture
testexamples_uni: info
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_C_X11_MPIUni  tree 
	-@echo "Completed compiling and running uniprocessor test examples"
	-@echo "========================================="

# Builds SLEPc test examples for a given architecture
testfortran_uni: info
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR FORTRAN EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@if [ "${FC}" != "" ]; then \
            ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=testexamples_Fortran_MPIUni  tree; \
            echo "Completed compiling and running uniprocessor fortran test examples"; \
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
	-@if [ "${SLEPC_INSTALL_DIR}" = "${SLEPC_DIR}" ]; then \
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
          if [ ! -d ${SLEPC_INSTALL_DIR}/include/finclude ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR}/include/finclude ; \
          fi;\
          cp -f include/finclude/*.h ${SLEPC_INSTALL_DIR}/include/finclude;\
          if [ ! -d ${SLEPC_INSTALL_DIR}/bmake ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR}/bmake ; \
          fi;\
          cp -f bmake/slepc_common* ${SLEPC_INSTALL_DIR}/bmake;\
          if [ ! -d ${SLEPC_INSTALL_DIR}/bmake/${PETSC_ARCH} ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR}/bmake/${PETSC_ARCH} ; \
          fi;\
          cp -f bmake/${PETSC_ARCH}/slepcconf ${SLEPC_INSTALL_DIR}/bmake/${PETSC_ARCH};\
          if [ ! -d ${SLEPC_INSTALL_DIR}/lib ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR}/lib ; \
          fi;\
          if [ ! -d ${SLEPC_INSTALL_DIR}/lib/${PETSC_ARCH} ]; then \
	    ${MKDIR} ${SLEPC_INSTALL_DIR}/lib/${PETSC_ARCH} ; \
          fi;\
          if [ -d lib/${PETSC_ARCH} ]; then \
            cp -f lib/${PETSC_ARCH}/* ${SLEPC_INSTALL_DIR}/lib/${PETSC_ARCH};\
            ${RANLIB} ${SLEPC_INSTALL_DIR}/lib/${PETSC_ARCH}/*.a ;\
            ${OMAKE} PETSC_ARCH=${PETSC_ARCH} SLEPC_DIR=${SLEPC_INSTALL_DIR} shared; \
          fi;\
          echo "sh/bash: SLEPC_DIR="${SLEPC_INSTALL_DIR}"; export SLEPC_DIR";\
          echo "csh/tcsh: setenv SLEPC_DIR "${SLEPC_INSTALL_DIR} ;\
          echo "Then do make test to verify correct install";\
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
	-${PETSC_DIR}/maint/wwwindex.py ${SLEPC_DIR} ${LOC}
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
# Some macros to check if the fortran interface is up-to-date.
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
	-@echo -------------- Functions missing in the fortran interface ---------------------
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

