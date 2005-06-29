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
	-@echo "config/configure.py run at " ${CONFIGURE_RUN_TIME}
	-@echo "config/configure.py options " ${CONFIGURE_OPTIONS}
	-@echo "-----------------------------------------"
	-@echo "Using SLEPc directory: ${SLEPC_DIR}"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "-----------------------------------------"
	-@grep "define SLEPC_VERSION" ${SLEPC_DIR}/include/slepcversion.h | ${SED} "s/........//"
	-@echo "-----------------------------------------"
	-@grep "define PETSC_VERSION" ${PETSC_DIR}/include/petscversion.h | ${SED} "s/........//"
	-@echo "-----------------------------------------"
	-@echo "Using configuration flags:"
	-@grep "\#define " ${PETSC_DIR}/bmake/${PETSC_ARCH}/petscconf.h
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${SLEPC_INCLUDE} ${PETSC_INCLUDE}"
	-@echo "Using PETSc/SLEPc flags: ${PETSCFLAGS} ${PCONF}"
	-@echo "------------------------------------------"
	-@echo "Using C/C++ compiler: ${CC} ${COPTFLAGS} ${CPPFLAGS}"
	-@echo "C/C++ Compiler version: " `${CCV}`
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran compiler: ${FC} ${FOPTFLAGS} ${FPPFLAGS}";\
	   echo "Fortran Compiler version: " `${FCV}`;\
         fi
	-@echo "-----------------------------------------"
	-@echo "Using C/C++ linker: ${CC_LINKER}"
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


# ------------------------------------------------------------------
#
# All remaining actions are intended for SLEPc developers only.
# SLEPc users should not generally need to use these commands.
#

chk_loc:
	@if [ ${LOC}foo = foo ] ; then \
	  echo "*********************** ERROR ************************" ; \
	  echo " Please specify LOC variable for eg: make allmanualpages LOC=/sandbox/petsc"; \
	  echo "******************************************************";  false; fi

chk_concepts_dir: chk_loc
	@if [ ! -d "${LOC}/docs/manualpages/concepts" ]; then \
	  echo Making directory ${LOC}/docs/manualpages/concepts for library; ${MKDIR} ${LOC}/docs/manualpages/concepts; fi

# Builds all the documentation
alldoc: alldoc1 alldoc2

# Build everything that goes into 'doc' dir except html sources
alldoc1: chk_loc deletemanualpages chk_concepts_dir
	-${OMAKE} ACTION=manualpages_buildcite tree_basic LOC=${LOC}
	-@sed -e s%man+../%man+manualpages/% ${LOC}/docs/manualpages/manualpages.cit > ${LOC}/docs/manualpages/htmlmap
	-@cat ${PETSC_DIR}/src/docs/mpi.www.index >> ${LOC}/docs/manualpages/htmlmap
#	cd src/docs/tex/manual; ${OMAKE} manual.pdf LOC=${LOC}
	-${OMAKE} ACTION=manualpages tree_basic LOC=${LOC}
	-${PETSC_DIR}/maint/wwwindex.py ${SLEPC_DIR} ${LOC}
	-${OMAKE} ACTION=manexamples tree_basic LOC=${LOC}
	-${OMAKE} manconcepts LOC=${LOC}
	-${OMAKE} ACTION=getexlist tree_basic LOC=${LOC}
	-${OMAKE} ACTION=exampleconcepts tree_basic LOC=${LOC}
	-@touch ${LOC}/docs/exampleconcepts
	-${PETSC_DIR}/maint/helpindex.py ${SLEPC_DIR} ${LOC}
#	-grep -h Polymorphic include/*.h | grep -v '#define ' | sed "s?PetscPolymorphic[a-zA-Z]*(??g" | cut -f1 -d"{" > tmppoly
#	-${PETSC_DIR}/maint/processpoly.py ${SLEPC_DIR} ${LOC}

# Builds .html versions of the source
# html overwrites some stuff created by update-docs - hence this is done later.
alldoc2: chk_loc
	-${OMAKE} ACTION=slepc_html PETSC_DIR=${PETSC_DIR} alltree LOC=${LOC}
	cp ${LOC}/docs/manual.htm ${LOC}/docs/index.html

# Deletes man pages (HTML version)
deletemanualpages: chk_loc
	-@if [ -d ${LOC} -a -d ${LOC}/docs/manualpages ]; then \
          find ${LOC}/docs/manualpages -type f -name "*.html" -exec ${RM} {} \; ;\
          ${RM} ${LOC}/docs/exampleconcepts ;\
          ${RM} ${LOC}/docs/manconcepts ;\
          ${RM} ${LOC}/docs/manualpages/manualpages.cit ;\
        fi

# Builds Fortran stub files
allfortranstubs:
	-@${PETSC_DIR}/maint/generatefortranstubs.py ${BFORT}

# -------------------------------------------------------------------------------
#
# Some macros to check if the fortran interface is up-to-date.
#
countfortranfunctions: 
	-@cd ${SLEPC_DIR}/src/fortran; egrep '^void' custom/*.c auto/*.c | \
	cut -d'(' -f1 | tr -s  ' ' | cut -d' ' -f2 | uniq | egrep -v "(^$$|Petsc)" | \
	sed "s/_$$//" | sort > /tmp/countfortranfunctions

countcfunctions:
	-@ grep extern ${SLEPC_DIR}/include/*.h *.h | grep "(" | tr -s ' ' | \
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
	-@cd ${SLEPC_DIR}/src/fortran/auto; grep '^void' *.c | grep 'MPI_Comm' | \
	tr -s ' ' | tr -s ':' ' ' |cut -d'(' -f1 | cut -d' ' -f1,3
	-@echo "========================================="
	-@echo "Functions with a String as an Argument"
	-@echo "========================================="
	-@cd ${SLEPC_DIR}/src/fortran/auto; grep '^void' *.c | grep 'char \*' | \
	tr -s ' ' | tr -s ':' ' ' |cut -d'(' -f1 | cut -d' ' -f1,3
	-@echo "========================================="
	-@echo "Functions with Pointers to PETSc Objects as Argument"
	-@echo "========================================="
	-@cd ${SLEPC_DIR}/src/fortran/auto; \
	_p_OBJ=`grep _p_ ${SLEPC_DIR}/include/*.h | tr -s ' ' | \
	cut -d' ' -f 3 | tr -s '\012' | grep -v '{' | cut -d'*' -f1 | \
	sed "s/_p_//g" | tr -s '\012 ' ' *|' ` ; \
	for OBJ in $$_p_OBJ; do \
	grep "$$OBJ \*" *.c | tr -s ' ' | tr -s ':' ' ' | \
	cut -d'(' -f1 | cut -d' ' -f1,3; \
	done 

