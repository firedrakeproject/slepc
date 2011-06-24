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

#include <slepcsys.h>
#include <private/epsimpl.h>          /*I "slepceps.h" I*/
#include <private/stimpl.h>           /*I "slepcst.h" I*/
#include <private/svdimpl.h>          /*I "slepcsvd.h" I*/
#include <private/qepimpl.h>          /*I "slepcqep.h" I*/
#include <private/ipimpl.h>           /*I "slepcip.h" I*/
#include <stdlib.h>
#include <slepcvec.h>

#undef __FUNCT__  
#define __FUNCT__ "SlepcPrintVersion"
/*
   SlepcPrintVersion - Prints SLEPc version info.

   Collective on MPI_Comm
*/
PetscErrorCode SlepcPrintVersion(MPI_Comm comm)
{
  PetscErrorCode ierr;
  char           version[256];
  
  PetscFunctionBegin;
  ierr = SlepcGetVersion(version,256);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"--------------------------------------------------------------------------\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"%s\n",version);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,SLEPC_AUTHOR_INFO);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"See docs/manual.html for help.\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"SLEPc libraries linked from %s\n",SLEPC_LIB_DIR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcPrintHelpIntro"
/*
   SlepcPrintHelpIntro - Prints introductory SLEPc help info.

   Collective on MPI_Comm
*/
PetscErrorCode SlepcPrintHelpIntro(MPI_Comm comm)
{
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(comm,"SLEPc help information includes that for the PETSc libraries, which provide\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"low-level system infrastructure and linear algebra tools.\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"--------------------------------------------------------------------------\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------Nasty global variables -------------------------------*/
/*
   Indicates whether SLEPc started PETSc, or whether it was 
   already started before SLEPc was initialized.
*/
PetscBool SlepcBeganPetsc = PETSC_FALSE; 
PetscBool SlepcInitializeCalled = PETSC_FALSE;
extern PetscLogEvent SLEPC_UpdateVectors,SLEPC_VecMAXPBY,SLEPC_SlepcDenseMatProd,SLEPC_SlepcDenseOrth,
                     SLEPC_SlepcDenseMatInvProd,SLEPC_SlepcDenseNorm,SLEPC_SlepcDenseCopy,SLEPC_VecsMult;

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
extern PetscDLLibrary DLLibrariesLoaded;

#undef __FUNCT__  
#define __FUNCT__ "SlepcInitialize_DynamicLibraries"
/*
    SlepcInitialize_DynamicLibraries - Adds the default dynamic link libraries to the 
    search path.
*/ 
PetscErrorCode SlepcInitialize_DynamicLibraries(void)
{
  PetscErrorCode ierr;
  PetscBool      found;
  char           libs[PETSC_MAX_PATH_LEN],dlib[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscStrcpy(libs,SLEPC_LIB_DIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libslepc");CHKERRQ(ierr);
  ierr = PetscDLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,&found);CHKERRQ(ierr);
  if (found) {
    ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate SLEPc dynamic library\n You cannot move the dynamic libraries!");
  }
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "SlepcInitialize_Packages"
/*
    SlepcInitialize_Packages - Initialize all SLEPc packages at the initialization.
*/ 
PetscErrorCode SlepcInitialize_Packages(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = STInitializePackage(PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSInitializePackage(PETSC_NULL);CHKERRQ(ierr);
  ierr = SVDInitializePackage(PETSC_NULL);CHKERRQ(ierr);
  ierr = QEPInitializePackage(PETSC_NULL);CHKERRQ(ierr);
  ierr = IPInitializePackage(PETSC_NULL);CHKERRQ(ierr);
  /* New special type of Vec, implemented in SLEPc */
  ierr = VecRegister_Comp(PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcInitialize_LogEvents"
/*
    SlepcInitialize_LogEvents - Initialize log events not pertaining to any object class.
*/ 
PetscErrorCode SlepcInitialize_LogEvents(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventRegister("UpdateVectors",0,&SLEPC_UpdateVectors);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMAXPBY",0,&SLEPC_VecMAXPBY);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseMatProd",EPS_CLASSID,&SLEPC_SlepcDenseMatProd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseOrth",EPS_CLASSID,&SLEPC_SlepcDenseOrth);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseMatInvProd",EPS_CLASSID,&SLEPC_SlepcDenseMatInvProd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseMatNorm",EPS_CLASSID,&SLEPC_SlepcDenseNorm);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DenseCopy",EPS_CLASSID,&SLEPC_SlepcDenseCopy);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecsMult",EPS_CLASSID,&SLEPC_VecsMult);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

.seealso: SlepcFinalize(), PetscInitialize()
@*/
PetscErrorCode SlepcInitialize(int *argc,char ***args,const char file[],const char help[])
{
  PetscErrorCode ierr,info=0;
  PetscBool      flg;

  PetscFunctionBegin;
  if (SlepcInitializeCalled) {
    PetscFunctionReturn(0); 
  }
  info = PetscSetHelpVersionFunctions(SlepcPrintHelpIntro,SlepcPrintVersion);CHKERRQ(info);
  ierr = PetscInitialized(&flg);CHKERRQ(ierr);
  if (!flg) {
    info = PetscInitialize(argc,args,file,help);CHKERRQ(info);
    SlepcBeganPetsc = PETSC_TRUE;
  }

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = SlepcInitialize_DynamicLibraries();CHKERRQ(ierr);
#else
  ierr = SlepcInitialize_Packages();CHKERRQ(ierr);
#endif
  ierr = SlepcInitialize_LogEvents();CHKERRQ(ierr);

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

#undef __FUNCT__  
#define __FUNCT__ "SlepcInitialized"
/*@
   SlepcInitialized - Determine whether SLEPc is initialized.
  
   Level: beginner

.seealso: SlepcInitialize(), SlepcInitializeFortran()
@*/
PetscErrorCode SlepcInitialized(PetscBool *isInitialized)
{
  PetscFunctionBegin;
  PetscValidPointer(isInitialized,1);
  *isInitialized = SlepcInitializeCalled;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
extern PetscBool PetscBeganMPI;

#undef __FUNCT__  
#define __FUNCT__ "SlepcInitializeMatlab"
/*
   SlepcInitializeMatlab - Calls SlepcInitialize() from MATLAB (analogue to
   PetscInitializeMatlab).

   Collective
  
   Level: advanced

.seealso: SlepcInitialize()
*/
PetscErrorCode SlepcInitializeMatlab(int argc,char **args,const char *filename,const char *help)
{
  PetscErrorCode ierr;
  int            myargc = argc;
  char           **myargs = args;

  PetscFunctionBegin;
  ierr = SlepcInitialize(&myargc,&myargs,filename,help);
  ierr = PetscPopSignalHandler();CHKERRQ(ierr);
  PetscBeganMPI = PETSC_FALSE;
  PetscFunctionReturn(ierr);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcInitializedMatlab"
/*
   SlepcInitializedMatlab - Has SLEPc been initialized already?

   Not Collective
  
   Level: advanced

   Notes: this is called only by the SLEPc MATLAB interface.

.seealso: SlepcInitialize()
*/
int SlepcInitializedMatlab(void)
{
  PetscBool flg;

  SlepcInitialized(&flg);
  if (flg) return 1;
  else return 0;
}
#endif

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
  ierr = SlepcInitialize_Packages();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
