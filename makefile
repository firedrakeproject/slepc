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
	-@${MAKE} slepc_all_build 2>&1 | tee make_log_${PETSC_ARCH}_${BOPT}
slepc_all_build: chk_slepc_dir info slepc_chklib_dir slepc_deletelibs slepc_build_c slepc_build_fortran slepc_shared
#
# Prints information about the system and version of SLEPc being compiled
#
info:
	-@echo "=========================================="
	-@echo On `date` on `hostname`
	-@echo Machine characteristics: `uname -a`
	-@echo "-----------------------------------------"
	-@echo "Using C compiler: ${C_CC} ${COPTFLAGS} ${CCPPFLAGS}"
	-@echo "C Compiler version: " `${C_CCV}`
	-@echo "Using C++ compiler: ${CXX_CC} ${COPTFLAGS} ${CCPPFLAGS}"
	-@echo "C++ Compiler version: " `${CXX_CCV}`
	-@echo "Using Fortran compiler: ${C_FC} ${FOPTFLAGS} ${FCPPFLAGS}"
	-@echo "Fortran Compiler version: " `${C_FCV}`
	-@echo "-----------------------------------------"
	-@grep SLEPC_VERSION_NUMBER ${SLEPC_DIR}/include/slepcversion.h | ${SED} "s/........//"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc/SLEPc flags: ${PETSCFLAGS} ${PCONF}"
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${SLEPC_INCLUDE}"
	-@echo "-----------------------------------------"
	-@echo "Using SLEPc directory: ${SLEPC_DIR}"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "------------------------------------------"
	-@echo "Using C linker: ${CLINKER}"
	-@echo "Using Fortran linker: ${FLINKER}"
	-@echo "Using libraries: ${SLEPC_LIB}"
	-@echo "=========================================="

#
# Builds the SLEPc libraries
#
slepc_build_c:
	-@echo "BEGINNING TO COMPILE SLEPc LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=libfast  tree 
	${RANLIB} ${SLEPC_LIB_DIR}/*.${LIB_SUFFIX}
	-@echo "Completed building SLEPc libraries"
	-@echo "========================================="

#
# Builds SLEPc Fortran source
#
slepc_build_fortran:
	-@echo "BEGINNING TO COMPILE SLEPc FORTRAN SOURCE"
	-@echo "========================================="
	-@cd src/fortran/custom; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} libf clean 
	${RANLIB} ${SLEPC_LIB_DIR}/libslepcfortran.a
	${RANLIB} ${SLEPC_LIB_DIR}/libslepc.a
	-@echo "Completed compiling SLEPc Fortran source"
	-@echo "========================================="

# Builds SLEPc test examples for a given BOPT and architecture
slepc_testexamples: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN SLEPc TEST EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_1  tree 
	-@echo "Completed compiling and running test examples"
	-@echo "========================================="

# Builds SLEPc test examples for a given BOPT and architecture
slepc_testfortran: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN SLEPc FORTRAN TEST EXAMPLES"
	-@echo "========================================="
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines or the way Fortran formats numbers"
	-@echo "some of the results may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_3  tree 
	-@echo "Completed compiling and running Fortran test examples"
	-@echo "========================================="

# Builds SLEPc test examples for a given BOPT and architecture
slepc_testexamples_uni: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_4  tree 
	-@echo "Completed compiling and running uniprocessor test examples"
	-@echo "========================================="
slepc_testfortran_uni: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR FORTRAN EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_9  tree 
	-@echo "Completed compiling and running uniprocessor fortran test examples"
	-@echo "========================================="

# Ranlib on the libraries
slepc_ranlib:
	${RANLIB} ${SLEPC_LIB_DIR}/*.${LIB_SUFFIX}

# Deletes SLEPc libraries
slepc_deletelibs: chkopts_basic
	-${RM} -f ${SLEPC_LIB_DIR}/*

slepc_shared: shared
slepc_chklib_dir: chklib_dir

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
slepc_alldoc: slepc_allmanualpages
#	cd docs/tex; ${OMAKE} ps  

# Deletes man pages (HTML version)
slepc_deletemanualpages:
	find ${LOC}/docs/manualpages -type f -name "*.html" -exec ${RM} {} \;
	${RM} ${LOC}/docs/exampleconcepts
	${RM} ${LOC}/docs/manconcepts
	${RM} ${LOC}/docs/manualpages/manualpages.cit
#	-${PETSC_DIR}/maint/update-docs.py ${SLEPC_DIR} ${LOC} clean

# Builds all versions of the man pages
slepc_allmanualpages: chk_loc slepc_deletemanualpages chk_concepts_dir
	-${OMAKE} ACTION=manualpages_buildcite tree_basic LOC=${LOC}
	-${OMAKE} ACTION=manualpages tree_basic LOC=${LOC}
	-${PETSC_DIR}/maint/wwwindex.py ${SLEPC_DIR} ${LOC}
	-${OMAKE} ACTION=slepc_manexamples tree LOC=${LOC}
	-${OMAKE} manconcepts LOC=${LOC}
	-${OMAKE} ACTION=getexlist tree LOC=${LOC}
	-${OMAKE} ACTION=slepc_exampleconcepts tree LOC=${LOC}
	-${PETSC_DIR}/maint/helpindex.py ${SLEPC_DIR} ${LOC}
	-${OMAKE} ACTION=slepc_html alltree LOC=${LOC}
#	-${PETSC_DIR}/maint/update-docs.py ${LOC}
	cp ${LOC}/docs/manual.htm ${LOC}/docs/index.html

# Builds Fortran stub files
slepc_allfortranstubs:
#	-@${RM} -f src/fortran/auto/*.c
#	-${OMAKE} ACTION=slepc_fortranstubs tree
#	-@cd src/fortran/auto; ${OMAKE} -f makefile slepc_fixfortran
	-@which ${BFORT} > /dev/null 2>&1;  \
        if [ "$$?" != "0" ]; then \
          echo "No bfort available, skipping building Fortran stubs";\
        else \
          ${RM} -f ${SLEPC_DIR}/src/fortran/auto/*.c ;\
	  touch ${SLEPC_DIR}/src/fortran/auto/makefile.src ;\
	  ${OMAKE} ACTION=slepc_fortranstubs tree_basic ;\
	  cd ${SLEPC_DIR}/src/fortran/auto; ${RM} makefile.src; echo SOURCEC = ` ls *.c | tr -s '\n' ' '` > makefile.src ;\
	  cd ${SLEPC_DIR}/src/fortran/auto; ${OMAKE} fixfortran ;\
        fi

# -------------------------------------------------------------------------------
#
# Some macros to check if the fortran interface is up-to-date.
#
slepc_countfortranfunctions: 
	-@cd ${SLEPC_DIR}/src/fortran; egrep '^void' custom/*.c auto/*.c | \
	cut -d'(' -f1 | tr -s  ' ' | cut -d' ' -f2 | uniq | egrep -v "(^$$|Tao)" | \
	sed "s/_$$//" | sort > /tmp/countfortranfunctions

slepc_countcfunctions:
	-@ grep -s extern ${SLEPC_DIR}/include/*.h *.h | grep "(" | tr -s ' ' | \
	cut -d'(' -f1 | cut -d' ' -f3 | grep -v "\*" | tr -s '\012' |  \
	tr 'A-Z' 'a-z' |  sort > /tmp/countcfunctions

slepc_difffortranfunctions: slepc_countfortranfunctions slepc_countcfunctions
	-@echo -------------- Functions missing in the Fortran interface ---------------------
	-@diff /tmp/countcfunctions /tmp/countfortranfunctions | grep "^<" | cut -d' ' -f2
	-@echo ----------------- Functions missing in the C interface ------------------------
	-@diff /tmp/countcfunctions /tmp/countfortranfunctions | grep "^>" | cut -d' ' -f2
	-@${RM}  /tmp/countcfunctions /tmp/countfortranfunctions

slepc_checkbadfortranstubs:
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
	-@echo "Functions with Pointers to SLEPc Objects as Argument"
	-@echo "========================================="
	-@cd ${SLEPC_DIR}/src/fortran/auto; \
	_p_OBJ=`grep _p_ ${SLEPC_DIR}/include/*.h | tr -s ' ' | \
	cut -d' ' -f 3 | tr -s '\012' | grep -v '{' | cut -d'*' -f1 | \
	sed "s/_p_//g" | tr -s '\012 ' ' *|' ` ; \
	for OBJ in $$_p_OBJ; do \
	grep "$$OBJ \*" *.c | tr -s ' ' | tr -s ':' ' ' | \
	cut -d'(' -f1 | cut -d' ' -f1,3; \
	done 

