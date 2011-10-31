#
# This is the makefile for installing SLEPc. See the Users Manual 
# for directions on installing SLEPc.
#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain
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

ALL: all
LOCDIR = .
DIRS   = src include docs 

# Include the rest of makefiles
include ${SLEPC_DIR}/conf/slepc_common

#
# Basic targets to build SLEPc library
all:
	@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} chkpetsc_dir
	@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} chkslepc_dir
	@if [ "${SLEPC_BUILD_USING_CMAKE}" != "" ]; then \
	   echo "=========================================="; \
           echo "Building SLEPc using CMake with ${MAKE_NP} build threads"; \
	   echo "Using SLEPC_DIR=${SLEPC_DIR}, PETSC_DIR=${PETSC_DIR} and PETSC_ARCH=${PETSC_ARCH}"; \
	   echo "=========================================="; \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} all-cmake; \
	 else \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} all-legacy; \
	 fi
	-@egrep -i "( error | error: |no such file or directory)" ${PETSC_ARCH}/conf/make.log > /dev/null; if [ "$$?" = "0" ]; then \
           echo "********************************************************************"; \
           echo "  Error during compile, check ${PETSC_ARCH}/conf/make.log"; \
           echo "  Send all contents of ${PETSC_ARCH}/conf to slepc-maint@grycap.upv.es";\
           echo "********************************************************************"; \
           exit 1; \
	 elif [ "${SLEPC_DESTDIR}" = "${SLEPC_DIR}/${PETSC_ARCH}" ]; then \
           echo "Now to check if the library is working do: make test";\
           echo "=========================================";\
	 else \
	   echo "Now to install the library do:";\
	   echo "make SLEPC_DIR=${PWD} PETSC_DIR=${PETSC_DIR} PETSC_ARCH=arch-installed-petsc install";\
	   echo "=========================================";\
	 fi

all-cmake:
	@${OMAKE} -j ${MAKE_NP} -C ${PETSC_ARCH} VERBOSE=1 2>&1 | tee ${PETSC_ARCH}/conf/make.log \
	          | egrep -v '( --check-build-system |cmake -E | -o CMakeFiles/slepc[[:lower:]]*.dir/| -o lib/libslepc|CMakeFiles/slepc[[:lower:]]*\.dir/(build|depend|requires)|-f CMakeFiles/Makefile2|Dependee .* is newer than depender |provides\.build. is up to date)'

all-legacy:
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} all_build 2>&1 | tee ${PETSC_ARCH}/conf/make.log

all_build: chk_petsc_dir chk_slepc_dir chklib_dir info deletelibs deletemods build shared_nomesg slepc4py_noinstall
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
	-@echo "Using PETSc configure options:" ${CONFIGURE_OPTIONS}
	-@echo "Using SLEPc configuration flags:"
	-@cat ${SLEPC_DIR}/${PETSC_ARCH}/conf/slepcvariables
	-@grep "\#define " ${SLEPC_DIR}/${PETSC_ARCH}/include/slepcconf.h
	-@echo "Using PETSc configuration flags:"
	-@if [ "${PETSC_ARCH}" != "arch-installed-petsc" ]; then \
	   grep "\#define " ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h; \
	 else \
	   grep "\#define " ${PETSC_DIR}/include/petscconf.h; \
         fi
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${SLEPC_INCLUDE} ${PETSC_CC_INCLUDES}"
	-@echo "------------------------------------------"
	-@echo "Using C/C++ compiler: ${PCC} ${PCC_FLAGS} ${COPTFLAGS} ${CFLAGS}"
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran include/module paths: ${PETSC_FC_INCLUDES}";\
	   echo "Using Fortran compiler: ${FC} ${FC_FLAGS} ${FFLAGS} ${FPP_FLAGS}";\
         fi
	-@echo "-----------------------------------------"
	-@echo "Using C/C++ linker: ${PCC_LINKER}"
	-@echo "Using C/C++ flags: ${PCC_LINKER_FLAGS}"
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran linker: ${FC_LINKER}";\
	   echo "Using Fortran flags: ${FC_LINKER_FLAGS}";\
         fi
	-@echo "-----------------------------------------"
	-@echo "Using library: ${SLEPC_LIB}"
	-@echo "------------------------------------------"
	-@echo "Using mpiexec: ${MPIEXEC}"
	-@echo "=========================================="

#
# Builds the SLEPc library
#
build:
	-@echo "BEGINNING TO COMPILE LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} ACTION=libfast slepc_tree 
	-@${RANLIB} ${SLEPC_LIB_DIR}/*.${AR_LIB_SUFFIX}  > tmpf 2>&1 ; ${GREP} -v "has no symbols" tmpf; ${RM} tmpf;
	-@echo "Completed building libraries"
	-@echo "========================================="

# Simple test examples for checking a correct installation
test: 
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} test_build 2>&1 | tee ./${PETSC_ARCH}/conf/test.log
test_build: 
	-@echo "Running test examples to verify correct installation"
	-@echo "Using SLEPC_DIR=${SLEPC_DIR}, PETSC_DIR=${PETSC_DIR} and PETSC_ARCH=${PETSC_ARCH}"
	@cd src/eps/examples/tests; ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} testtest10
	@if [ "${FC}" != "" ]; then cd src/eps/examples/tests; ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} testtest7f; fi;
	-@if [ "${BLOPEX_LIB}" != "" ]; then cd src/eps/examples/tests; ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} testtest5_blopex; fi; 
	-@echo "Completed test examples"

# Builds SLEPc test examples for C
testexamples: info
	-@echo "BEGINNING TO COMPILE AND RUN SLEPc TEST EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} \
	   ACTION=testexamples_C slepc_tree 
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
	    ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} \
	      ACTION=testexamples_Fortran slepc_tree ; \
            echo "Completed compiling and running Fortran test examples"; \
          else \
            echo "Error: No FORTRAN compiler available"; \
          fi
	-@echo "========================================="

# Test BLOPEX use
testblopex:
	-@echo "BEGINNING TO COMPILE AND RUN SLEPc BLOPEX TEST EXAMPLES"
	-@echo "========================================="
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the results may not match exactly."
	-@echo "========================================="
	-@if [ "${BLOPEX_LIB}" != "" ]; then \
	    ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} \
	      ACTION=testexamples_BLOPEX slepc_tree ; \
            echo "Completed compiling and running BLOPEX test examples"; \
          else \
            echo "Error: SLEPc has not been configured with BLOPEX"; \
          fi
	-@echo "========================================="

# Ranlib on the library
ranlib:
	${RANLIB} ${SLEPC_LIB_DIR}/*.${AR_LIB_SUFFIX}

# Deletes SLEPc library
deletelibs:
	-${RM} -r ${SLEPC_LIB_DIR}/libslepc*.*
deletemods:
	-${RM} -f ${SLEPC_DIR}/${PETSC_ARCH}/include/slepc*.mod

# Cleans up build
allclean: deletelibs deletemods
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} ACTION=clean slepc_tree

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
#
# Install relevant files in the prefix directory
#
install:
	-@if [ "${PETSC_ARCH}" = "" ]; then \
	  echo "PETSC_ARCH is undefined";\
	elif [ "${SLEPC_DESTDIR}" = "${SLEPC_DIR}/${PETSC_ARCH}" ]; then \
	  echo "Install directory is current directory; nothing needs to be done";\
        else \
	  echo Installing SLEPc at ${SLEPC_DESTDIR};\
          if [ ! -d `dirname ${SLEPC_DESTDIR}` ]; then \
	    ${MKDIR} `dirname ${SLEPC_DESTDIR}` ; \
          fi;\
          if [ ! -d ${SLEPC_DESTDIR} ]; then \
	    ${MKDIR} ${SLEPC_DESTDIR} ; \
          fi;\
          if [ ! -d ${SLEPC_DESTDIR}/include ]; then \
	    ${MKDIR} ${SLEPC_DESTDIR}/include ; \
          fi;\
          cp -f include/*.h ${SLEPC_DESTDIR}/include;\
          cp -f ${PETSC_ARCH}/include/*.h ${SLEPC_DESTDIR}/include;\
          if [ -f ${PETSC_ARCH}/include/slepceps.mod ]; then \
            cp -f ${PETSC_ARCH}/include/*.mod ${SLEPC_DESTDIR}/include;\
          fi;\
          if [ ! -d ${SLEPC_DESTDIR}/include/finclude ]; then \
	    ${MKDIR} ${SLEPC_DESTDIR}/include/finclude ; \
          fi;\
          cp -f include/finclude/*.h* ${SLEPC_DESTDIR}/include/finclude;\
          if [ -d include/finclude/ftn-auto ]; then \
            if [ ! -d ${SLEPC_DESTDIR}/include/finclude/ftn-auto ]; then \
	      ${MKDIR} ${SLEPC_DESTDIR}/include/finclude/ftn-auto ; \
            fi;\
            cp -f include/finclude/ftn-auto/*.h90 ${SLEPC_DESTDIR}/include/finclude/ftn-auto;\
          fi;\
          if [ ! -d ${SLEPC_DESTDIR}/include/finclude/ftn-custom ]; then \
	    ${MKDIR} ${SLEPC_DESTDIR}/include/finclude/ftn-custom ; \
          fi;\
          cp -f include/finclude/ftn-custom/*.h90 ${SLEPC_DESTDIR}/include/finclude/ftn-custom;\
          if [ ! -d ${SLEPC_DESTDIR}/include/private ]; then \
	    ${MKDIR} ${SLEPC_DESTDIR}/include/private ; \
          fi;\
          cp -f include/private/*.h ${SLEPC_DESTDIR}/include/private;\
          if [ ! -d ${SLEPC_DESTDIR}/conf ]; then \
	    ${MKDIR} ${SLEPC_DESTDIR}/conf ; \
          fi;\
          for dir in bin bin/matlab bin/matlab/classes bin/matlab/classes/examples \
              bin/matlab/classes/examples/tutorials; \
          do \
            if [ ! -d ${SLEPC_DESTDIR}/$$dir ]; then ${MKDIR} ${SLEPC_DESTDIR}/$$dir; fi;\
          done; \
          for dir in bin/matlab/classes bin/matlab/classes/examples/tutorials; \
          do \
            cp -f $$dir/*.m ${SLEPC_DESTDIR}/$$dir;\
          done; \
          cp -f bin/matlab/classes/slepcmatlabheader.h ${SLEPC_DESTDIR}/bin/matlab/classes;\
          cp -f conf/slepc_* ${SLEPC_DESTDIR}/conf;\
          cp -f ${PETSC_ARCH}/conf/slepcvariables ${SLEPC_DESTDIR}/conf;\
          cp -f ${PETSC_ARCH}/conf/slepcrules ${SLEPC_DESTDIR}/conf;\
          if [ ! -d ${SLEPC_DESTDIR}/lib ]; then \
	    ${MKDIR} ${SLEPC_DESTDIR}/lib ; \
          fi;\
          if [ -d ${PETSC_ARCH}/lib ]; then \
            cp -f ${PETSC_ARCH}/lib/*.${AR_LIB_SUFFIX} ${SLEPC_DESTDIR}/lib;\
            ${RANLIB} ${SLEPC_DESTDIR}/lib/*.${AR_LIB_SUFFIX} ;\
            ${OMAKE} PETSC_DIR=${PETSC_DIR} PETSC_ARCH="" SLEPC_DIR=${SLEPC_DESTDIR} OTHERSHAREDLIBS="${PETSC_LIB}" shared; \
            ${OMAKE} PETSC_DIR=${PETSC_DIR} PETSC_ARCH="" SLEPC_DIR=${SLEPC_DESTDIR} slepc4py; \
          fi;\
          echo "====================================";\
	  echo "Install complete.";\
	  echo "It is usable with SLEPC_DIR=${SLEPC_DESTDIR} PETSC_DIR=${PETSC_DIR} [and no more PETSC_ARCH].";\
          echo "Run the following to verify the install (in current directory):";\
          echo "make SLEPC_DIR=${SLEPC_DESTDIR} PETSC_DIR=${PETSC_DIR} test";\
          echo "====================================";\
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
	-${PYTHON} ${PETSC_DIR}/bin/maint/wwwindex.py ${SLEPC_DIR} ${LOC}
	-${OMAKE} ACTION=slepc_manexamples tree_basic LOC=${LOC}

# Builds .html versions of the source
alldoc2: chk_loc
	-${OMAKE} ACTION=slepc_html PETSC_DIR=${PETSC_DIR} alltree LOC=${LOC}
	cp ${LOC}/docs/manual.htm ${LOC}/docs/index.html

# Deletes documentation
alldocclean: deletemanualpages allcleanhtml
deletemanualpages: chk_loc
	-@if [ -d ${LOC} -a -d ${LOC}/docs/manualpages ]; then \
          find ${LOC}/docs/manualpages -type f -name "*.html" -exec ${RM} {} \; ;\
          ${RM} ${LOC}/docs/manualpages/manualpages.cit ;\
        fi
allcleanhtml: 
	-${OMAKE} ACTION=cleanhtml PETSC_DIR=${PETSC_DIR} alltree

# Builds Fortran stub files
allfortranstubs:
	-@${PYTHON} ${SLEPC_DIR}/config/generatefortranstubs.py ${BFORT}
deletefortranstubs:
	-@find . -type d -name ftn-auto | xargs rm -rf 

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
	-@ ls ${SLEPC_DIR}/include/*.h | grep -v slepcblaslapack.h | \
	xargs grep extern | grep "(" | tr -s ' ' | \
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

# Generate tags with PETSc's script
alletags:
	-@${PYTHON} ${PETSC_DIR}/bin/maint/generateetags.py
	-@find config -type f -name "*.py" |grep -v SCCS | xargs etags -o TAGS_PYTHON

