/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

   This file is part of SLEPc.
      
   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY 
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS 
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for 
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "slepcsys.h"           /*I "slepcsys.h" I*/
#include "private/epsimpl.h"
#include "private/stimpl.h"
#include "private/svdimpl.h"
#include "private/qepimpl.h"
#include "private/ipimpl.h"
#include <stdlib.h>

#undef __FUNCT__  
#define __FUNCT__ "SlepcPrintVersion"
/*
   SlepcPrintVersion - Prints SLEPc version info.

   Collective on MPI_Comm
*/
PetscErrorCode SlepcPrintVersion(MPI_Comm comm)
{
  PetscErrorCode  info = 0;
  
  PetscFunctionBegin;

  info = (*PetscHelpPrintf)(comm,"--------------------------------------------\
------------------------------\n"); CHKERRQ(info);
#if (PETSC_VERSION_RELEASE == 1)
  info = (*PetscHelpPrintf)(comm,"SLEPc Release Version %d.%d.%d-%d, %s\n",
#else
  info = (*PetscHelpPrintf)(comm,"SLEPc Development Version %d.%d.%d-%d, %s\n",
#endif
    SLEPC_VERSION_MAJOR,SLEPC_VERSION_MINOR,SLEPC_VERSION_SUBMINOR,SLEPC_VERSION_PATCH,SLEPC_VERSION_PATCH_DATE); CHKERRQ(info);
  info = (*PetscHelpPrintf)(comm,SLEPC_AUTHOR_INFO); CHKERRQ(info);
  info = (*PetscHelpPrintf)(comm,"See docs/manual.html for help. \n"); CHKERRQ(info);
#if !defined(PARCH_win32)
  info = (*PetscHelpPrintf)(comm,"SLEPc libraries linked from %s\n",SLEPC_LIB_DIR); CHKERRQ(info);
#endif
  info = (*PetscHelpPrintf)(comm,"--------------------------------------------\
------------------------------\n"); CHKERRQ(info);

  PetscFunctionReturn(info);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcPrintHelpIntro"
/*
   SlepcPrintHelpIntro - Prints introductory SLEPc help info.

   Collective on MPI_Comm
*/
PetscErrorCode SlepcPrintHelpIntro(MPI_Comm comm)
{
  PetscErrorCode  info = 0;
  
  PetscFunctionBegin;

  info = (*PetscHelpPrintf)(comm,"--------------------------------------------\
------------------------------\n"); CHKERRQ(info);
  info = (*PetscHelpPrintf)(comm,"SLEPc help information includes that for the PETSc libraries, which provide\n"); CHKERRQ(info);
  info = (*PetscHelpPrintf)(comm,"low-level system infrastructure and linear algebra tools.\n"); CHKERRQ(info);
  info = (*PetscHelpPrintf)(comm,"--------------------------------------------\
------------------------------\n"); CHKERRQ(info);

  PetscFunctionReturn(info);
}

/* ------------------------Nasty global variables -------------------------------*/
/*
   Indicates whether SLEPc started PETSc, or whether it was 
   already started before SLEPc was initialized.
*/
PetscTruth  SlepcBeganPetsc = PETSC_FALSE; 
PetscTruth  SlepcInitializeCalled = PETSC_FALSE;
extern PetscLogEvent SLEPC_UpdateVectors, SLEPC_VecMAXPBY, SLEPC_SlepcDenseMatProd, SLEPC_SlepcDenseOrth,
                     SLEPC_SlepcDenseMatInvProd, SLEPC_SlepcDenseNorm, SLEPC_SlepcDenseCopy, SLEPC_VecsMult;

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
extern PetscDLLibrary DLLibrariesLoaded;
#endif

#undef __FUNCT__  
#define __FUNCT__ "SlepcInitialize"
/*@C 
   SlepcInitialize - Initializes the SLEPc library. SlepcInitialize() calls
   PetscInitialize() if that has not been called yet, so this routine should
   always be called near the beginning of your program.

   Collective on MPI_COMM_WORLD or PETSC_COMM_WORLD if it has been set

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
.  file - [optional] PETSc database file, defaults to ~username/.petscrc
          (use PETSC_NULL for default)
-  help - [optional] Help message to print, use PETSC_NULL for no message

   Fortran Note:
   Fortran syntax is very similar to that of PetscInitialize()
   
   Level: beginner

.seealso: SlepcInitializeFortran(), SlepcFinalize(), PetscInitialize()
@*/
PetscErrorCode SlepcInitialize(int *argc,char ***args,char file[],const char help[])
{
  PetscErrorCode ierr;
  PetscErrorCode info=0;
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  char           libs[PETSC_MAX_PATH_LEN],dlib[PETSC_MAX_PATH_LEN];
  PetscTruth     found;
#endif

  PetscFunctionBegin;

  if (SlepcInitializeCalled) {
    PetscFunctionReturn(0); 
  }

#if !defined(PARCH_t3d)
  info = PetscSetHelpVersionFunctions(SlepcPrintHelpIntro,SlepcPrintVersion);CHKERRQ(info);
#endif

  if (!PetscInitializeCalled) {
    info = PetscInitialize(argc,args,file,help);CHKERRQ(info);
    SlepcBeganPetsc = PETSC_TRUE;
  }

  /*
      Load the dynamic libraries
  */

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = PetscStrcpy(libs,SLEPC_LIB_DIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libslepc");CHKERRQ(ierr);
  ierr = PetscDLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,&found);CHKERRQ(ierr);
  if (found) {
    ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Unable to locate SLEPc dynamic library %s \n",libs);
  }
#else
  ierr = STInitializePackage(PETSC_NULL); CHKERRQ(ierr);
  ierr = EPSInitializePackage(PETSC_NULL); CHKERRQ(ierr);
  ierr = SVDInitializePackage(PETSC_NULL); CHKERRQ(ierr);
  ierr = QEPInitializePackage(PETSC_NULL); CHKERRQ(ierr);
  ierr = IPInitializePackage(PETSC_NULL); CHKERRQ(ierr);
#endif

  ierr = PetscLogEventRegister("UpdateVectors",0,&SLEPC_UpdateVectors);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMAXPBY",0,&SLEPC_VecMAXPBY);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseMatProd", EPS_COOKIE, &SLEPC_SlepcDenseMatProd); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseOrth", EPS_COOKIE, &SLEPC_SlepcDenseOrth); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseMatInvProd", EPS_COOKIE, &SLEPC_SlepcDenseMatInvProd); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseMatNorm", EPS_COOKIE, &SLEPC_SlepcDenseNorm); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseCopy", EPS_COOKIE, &SLEPC_SlepcDenseCopy); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecsMult", EPS_COOKIE, &SLEPC_VecsMult); CHKERRQ(ierr);

#if defined(PETSC_HAVE_DRAND48)
  /* work-around for Cygwin drand48() initialization bug */
  srand48(0);
#endif

  SlepcInitializeCalled = PETSC_TRUE;
  PetscInfo(0,"SLEPc successfully started\n");
  PetscFunctionReturn(info);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcFinalize"
/*@
   SlepcFinalize - Checks for options to be called at the conclusion
   of the SLEPc program and calls PetscFinalize().

   Collective on PETSC_COMM_WORLD

   Level: beginner

.seealso: SlepcInitialize(), PetscFinalize()
@*/
PetscErrorCode SlepcFinalize(void)
{
  PetscErrorCode info=0;
  
  PetscFunctionBegin;
  PetscInfo(0,"SLEPc successfully ended!\n");

  if (SlepcBeganPetsc) {
    info = PetscFinalize();CHKERRQ(info);
  }

  SlepcInitializeCalled = PETSC_FALSE;

  PetscFunctionReturn(info);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister_slepc"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library 
  it is in is opened.

  This one registers all the EPS and ST methods in the libslepc.a
  library.

  Input Parameter:
  path - library path
 */
PetscErrorCode PetscDLLibraryRegister_slepc(char *path)
{
  PetscErrorCode ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;

  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = STInitializePackage(path); CHKERRQ(ierr);
  ierr = EPSInitializePackage(path); CHKERRQ(ierr);
  ierr = SVDInitializePackage(path); CHKERRQ(ierr);
  ierr = QEPInitializePackage(path); CHKERRQ(ierr);
  ierr = IPInitializePackage(path); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
