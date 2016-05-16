#
# This is the makefile for installing SLEPc. See the Users Manual
# for directions on installing SLEPc.
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

ALL: all
LOCDIR = .
DIRS   = src include docs

# Include the rest of makefiles
include ./${PETSC_ARCH}/lib/slepc/conf/slepcvariables
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

#
# Basic targets to build SLEPc library
all: chk_makej
	@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} chk_petscdir chk_slepcdir | tee ./${PETSC_ARCH}/lib/slepc/conf/make.log
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
	   ${OMAKE_PRINTDIR} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} all-gnumake-local 2>&1 | tee -a ./${PETSC_ARCH}/lib/slepc/conf/make.log; \
	elif [ "${SLEPC_BUILD_USING_CMAKE}" != "" ]; then \
	   if [ "${SLEPC_DESTDIR}" = "${SLEPC_DIR}/${PETSC_ARCH}" ]; then \
	     ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} cmakegen; \
	   fi; \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} all-cmake-local 2>&1 | tee ./${PETSC_ARCH}/lib/slepc/conf/make.log \
	          | egrep -v '( --check-build-system |cmake -E | -o CMakeFiles/slepc[[:lower:]]*.dir/| -o lib/libslepc|CMakeFiles/slepc[[:lower:]]*\.dir/(build|depend|requires)|-f CMakeFiles/Makefile2|Dependee .* is newer than depender |provides\.build. is up to date)'; \
	 else \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} all-legacy-local 2>&1 | tee ./${PETSC_ARCH}/lib/slepc/conf/make.log | ${GREP} -v "has no symbols"; \
	 fi
	@egrep -i "( error | error: |no such file or directory)" ${PETSC_ARCH}/lib/slepc/conf/make.log | tee ./${PETSC_ARCH}/lib/slepc/conf/error.log > /dev/null
	@if test -s ./${PETSC_ARCH}/lib/slepc/conf/error.log; then \
           printf ${PETSC_TEXT_HILIGHT}"*******************************ERROR************************************\n" 2>&1 | tee -a ./${PETSC_ARCH}/lib/slepc/conf/make.log; \
           echo "  Error during compile, check ./${PETSC_ARCH}/lib/slepc/conf/make.log" 2>&1 | tee -a ./${PETSC_ARCH}/lib/slepc/conf/make.log; \
           echo "  Send all contents of ./${PETSC_ARCH}/lib/slepc/conf to slepc-maint@upv.es" 2>&1 | tee -a ./${PETSC_ARCH}/lib/slepc/conf/make.log;\
           printf "************************************************************************"${PETSC_TEXT_NORMAL}"\n" 2>&1 | tee -a ./${PETSC_ARCH}/lib/slepc/conf/make.log; \
	 elif [ "${SLEPC_DESTDIR}" = "${SLEPC_DIR}/${PETSC_ARCH}" ]; then \
           echo "Now to check if the library is working do: make test";\
           echo "=========================================";\
	 else \
	   echo "Now to install the library do:";\
	   echo "make SLEPC_DIR=${PWD} PETSC_DIR=${PETSC_DIR} install";\
	   echo "=========================================";\
	 fi
	@if test -s ./${PETSC_ARCH}/lib/slepc/conf/error.log; then exit 1; fi

cmakegen:
	-@${PYTHON} config/cmakegen.py

all-gnumake:
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
          ${OMAKE_PRINTDIR} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} SLEPC_BUILD_USING_CMAKE="" all;\
        else printf ${PETSC_TEXT_HILIGHT}"Build not configured for GNUMAKE. Quiting"${PETSC_TEXT_NORMAL}"\n"; exit 1; fi

all-cmake:
	@if [ "${SLEPC_BUILD_USING_CMAKE}" != "" ]; then \
	  if [ "${SLEPC_DESTDIR}" = "${SLEPC_DIR}/${PETSC_ARCH}" ]; then \
	    ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} cmakegen; \
	  fi; \
          ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} MAKE_IS_GNUMAKE="" all;\
        else printf ${PETSC_TEXT_HILIGHT}"Build not configured for CMAKE. Quiting"${PETSC_TEXT_NORMAL}"\n"; exit 1; fi

all-legacy:
	@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} SLEPC_BUILD_USING_CMAKE="" MAKE_IS_GNUMAKE="" all

all-gnumake-local: chk_makej info slepc_gnumake

all-cmake-local: chk_makej info cmakegen slepc_cmake

all-legacy-local: chk_makej chk_petsc_dir chk_slepc_dir chklib_dir info deletelibs deletemods build slepc_shared
#
# Prints information about the system and version of SLEPc being compiled
#
info: chk_makej
	-@echo "=========================================="
	-@echo Starting on `hostname` at `date`
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
	-@echo "Using PETSc configure options: ${CONFIGURE_OPTIONS}"
	-@echo "Using SLEPc configuration flags:"
	-@cat ${SLEPC_DIR}/${PETSC_ARCH}/lib/slepc/conf/slepcvariables
	-@grep "\#define " ${SLEPC_DIR}/${PETSC_ARCH}/include/slepcconf.h
	-@echo "Using PETSc configuration flags:"
	-@if [ "${INSTALLED_PETSC}" != "" ]; then \
	   grep "\#define " ${PETSC_DIR}/include/petscconf.h; \
	 else \
	   grep "\#define " ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h; \
         fi
	-@echo "-----------------------------------------"
	-@echo "Using C/C++ include paths: ${SLEPC_CC_INCLUDES}"
	-@echo "Using C/C++ compiler: ${PCC} ${PCC_FLAGS} ${COPTFLAGS} ${CFLAGS}"
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran include/module paths: ${SLEPC_FC_INCLUDES}";\
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
	-@echo "Using libraries: ${SLEPC_LIB}"
	-@echo "------------------------------------------"
	-@echo "Using mpiexec: ${MPIEXEC}"
	-@echo "=========================================="

#
# Builds the SLEPc library
#
build: chk_makej
	-@echo "BEGINNING TO COMPILE LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} ACTION=libfast slepc_tree
	-@${RANLIB} ${SLEPC_LIB_DIR}/*.${AR_LIB_SUFFIX}  > tmpf 2>&1 ; ${GREP} -v "has no symbols" tmpf; ${RM} tmpf;
	-@echo "Completed building libraries"
	-@echo "========================================="

# Simple test examples for checking a correct installation
check: test
test:
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} test_build 2>&1 | tee ./${PETSC_ARCH}/lib/slepc/conf/test.log
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
deletelibs: chk_makej
	-${RM} -r ${SLEPC_LIB_DIR}/libslepc*.*
deletemods: chk_makej
	-${RM} -f ${SLEPC_DIR}/${PETSC_ARCH}/include/slepc*.mod

# Cleans up build
allclean-legacy: deletelibs deletemods
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} ACTION=clean slepc_tree
allclean-cmake:
	-@cd ./${PETSC_ARCH} && ${OMAKE} clean
allclean-gnumake:
	-@${OMAKE} -f gmakefile clean

allclean:
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} allclean-gnumake; \
	elif [ "${PETSC_BUILD_USING_CMAKE}" != "" ]; then \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} allclean-cmake; \
	else \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} SLEPC_DIR=${SLEPC_DIR} allclean-legacy; \
	fi

clean:: allclean

#
# Check if PETSC_DIR variable specified is valid
#
chk_petsc_dir:
	@if [ ! -f ${PETSC_DIR}/include/petscversion.h ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "Incorrect PETSC_DIR specified: ${PETSC_DIR}!                             "; \
	  echo "You need to use / to separate directories, not \\!                       "; \
	  echo "Aborting build                                                           "; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
	  false; fi
#
# Check if SLEPC_DIR variable specified is valid
#
chk_slepc_dir:
	@if [ ! -f ${SLEPC_DIR}/include/slepcversion.h ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "Incorrect SLEPC_DIR specified: ${SLEPC_DIR}!                             "; \
	  echo "You need to use / to separate directories, not \\!                       "; \
	  echo "Aborting build                                                           "; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
	  false; fi
#
# Install relevant files in the prefix directory
#
install:
	-@${PYTHON} ./config/install.py ${SLEPC_DIR} ${PETSC_DIR} ${SLEPC_DESTDIR} ${PETSC_ARCH} ${AR_LIB_SUFFIX} ${RANLIB};

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

# modify all generated html files and add in version number, date, canonical URL info.
docsetdate: chk_petscdir
	@echo "Updating generated html files with slepc version, date, canonical URL info";\
        version_release=`grep '^#define SLEPC_VERSION_RELEASE ' include/slepcversion.h |tr -s ' ' | cut -d ' ' -f 3`; \
        version_major=`grep '^#define SLEPC_VERSION_MAJOR ' include/slepcversion.h |tr -s ' ' | cut -d ' ' -f 3`; \
        version_minor=`grep '^#define SLEPC_VERSION_MINOR ' include/slepcversion.h |tr -s ' ' | cut -d ' ' -f 3`; \
        version_subminor=`grep '^#define SLEPC_VERSION_SUBMINOR ' include/slepcversion.h |tr -s ' ' | cut -d ' ' -f 3`; \
        if  [ $${version_release} = 0 ]; then \
          slepcversion=slepc-dev; \
          export slepcversion; \
        elif [ $${version_release} = 1 ]; then \
          slepcversion=slepc-$${version_major}.$${version_minor}.$${version_subminor}; \
          export slepcversion; \
        else \
          echo "Unknown SLEPC_VERSION_RELEASE: $${version_release}"; \
          exit; \
        fi; \
        datestr=`git log -1 --pretty=format:%ci | cut -d ' ' -f 1`; \
        export datestr; \
        gitver=`git describe`; \
        export gitver; \
        find * -type d -wholename 'arch-*' -prune -o -type f -name \*.html \
          -exec perl -pi -e 's^(<body.*>)^$$1\n   <div id=\"version\" align=right><b>$$ENV{slepcversion} $$ENV{datestr}</b></div>\n   <div id="bugreport" align=right><a href="mailto:slepc-maint\@upv.es?subject=Typo or Error in Documentation &body=Please describe the typo or error in the documentation: $$ENV{slepcversion} $$ENV{gitver} {} "><small>Report Typos and Errors</small></a></div>^i' {} \; \
          -exec perl -pi -e 's^(<head>)^$$1 <link rel="canonical" href="http://slepc.upv.es/documentation/current/{}" />^i' {} \; ; \
        echo "Done fixing version number, date, canonical URL info"

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
	-@${RM} -rf include/slepc/finclude/ftn-auto/*-tmpdir
	-@${PYTHON} ${SLEPC_DIR}/bin/maint/generatefortranstubs.py ${BFORT}
	-@${PYTHON} ${SLEPC_DIR}/bin/maint/generatefortranstubs.py -merge  ${VERBOSE}
	-@${RM} -rf include/slepc/finclude/ftn-auto/*-tmpdir
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

# Generate tags
alletags:
	-@${PYTHON} ${SLEPC_DIR}/bin/maint/generateetags.py
	-@find config -type f -name "*.py" |grep -v SCCS | xargs etags -o TAGS_PYTHON

