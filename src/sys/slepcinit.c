
#include "slepc.h" /*I "slepc.h" I*/
#include "slepceps.h"
#include "slepcst.h"

#undef __FUNCT__  
#define __FUNCT__ "SlepcPrintVersion"
/*
   SlepcPrintVersion - Prints SLEPc version info.

   Collective on MPI_Comm
*/
PetscErrorCode SlepcPrintVersion(MPI_Comm comm)
{
  int  info = 0;
  
  PetscFunctionBegin;

  info = (*PetscHelpPrintf)(comm,"--------------------------------------------\
------------------------------\n"); CHKERRQ(info);
  info = (*PetscHelpPrintf)(comm,"\t   %s\n",SLEPC_VERSION_NUMBER); CHKERRQ(info);
  info = (*PetscHelpPrintf)(comm,"%s",SLEPC_AUTHOR_INFO); CHKERRQ(info);
  info = (*PetscHelpPrintf)(comm,"See docs/index.html for help. \n"); CHKERRQ(info);
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
  int  info = 0;
  
  PetscFunctionBegin;

  info = (*PetscHelpPrintf)(comm,"--------------------------------------------\
------------------------------\n"); CHKERRQ(info);
  info = (*PetscHelpPrintf)(comm,"SLEPc help information includes that for the PETSc libraries, which provide\n"); CHKERRQ(info);
  info = (*PetscHelpPrintf)(comm,"low-level system infrastructure and linear algebra tools.\n"); CHKERRQ(info);
  info = (*PetscHelpPrintf)(comm,"--------------------------------------------\
------------------------------\n"); CHKERRQ(info);

  PetscFunctionReturn(info);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcRegisterEvents"
PetscEvent EPS_SetUp, EPS_Solve, ST_SetUp, ST_Apply, ST_ApplyB, ST_ApplyNoB, EPS_Orthogonalization, ST_InnerProduct;

/*
   SlepcRegisterEvents - Registers SLEPc events for use in performance logging.
*/
PetscErrorCode SlepcRegisterEvents()
{
  int  info = 0;
  
  PetscFunctionBegin;

  info = PetscLogEventRegister(&EPS_SetUp,"EPSSetUp",PETSC_NULL); CHKERRQ(info);
  info = PetscLogEventRegister(&EPS_Solve,"EPSSolve",PETSC_NULL); CHKERRQ(info);
  info = PetscLogEventRegister(&ST_SetUp,"STSetUp",PETSC_NULL); CHKERRQ(info);
  info = PetscLogEventRegister(&ST_Apply,"STApply",PETSC_NULL); CHKERRQ(info);
  info = PetscLogEventRegister(&ST_ApplyB,"STApplyB",PETSC_NULL); CHKERRQ(info);
  info = PetscLogEventRegister(&ST_ApplyNoB,"STApplyNoB",PETSC_NULL); CHKERRQ(info);
  info = PetscLogEventRegister(&EPS_Orthogonalization,"EPSOrthogonalization",PETSC_NULL); CHKERRQ(info);
  info = PetscLogEventRegister(&ST_InnerProduct,"STInnerProduct",PETSC_NULL); CHKERRQ(info);

  PetscFunctionReturn(info);
}

/* ------------------------Nasty global variables -------------------------------*/
/*
   Indicates whether SLEPc started PETSc, or whether it was 
   already started before SLEPc was initialized.
*/
PetscTruth  SlepcBeganPetsc = PETSC_FALSE; 
PetscTruth  SlepcInitializeCalled = PETSC_FALSE;
PetscRandom rctx = PETSC_NULL;
PetscCookie EPS_COOKIE = 0;
PetscCookie ST_COOKIE = 0;

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
extern PetscDLLibraryList DLLibrariesLoaded;
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
  PetscErrorCode ierr,info=0;
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  char           libs[PETSC_MAX_PATH_LEN],dlib[PETSC_MAX_PATH_LEN];
  PetscTruth     found;
#endif

  PetscFunctionBegin;

  if (SlepcInitializeCalled==PETSC_TRUE) {
    PetscFunctionReturn(0); 
  }

#if !defined(PARCH_t3d)
  info = PetscSetHelpVersionFunctions(SlepcPrintHelpIntro,SlepcPrintVersion);CHKERRQ(info);
#endif

  if (!PetscInitializeCalled) {
    info = PetscInitialize(argc,args,file,help);CHKERRQ(info);
    SlepcBeganPetsc = PETSC_TRUE;
  }

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rctx);CHKERRQ(ierr);

  EPS_COOKIE = 0;
  ierr = PetscLogClassRegister(&EPS_COOKIE,"Eigenproblem Solver");CHKERRQ(ierr);
  ST_COOKIE = 0;
  ierr = PetscLogClassRegister(&ST_COOKIE,"Spectral Transform");CHKERRQ(ierr);

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
    SETERRQ1(1,"Unable to locate SLEPc dynamic library %s \n You cannot move the dynamic libraries!\n or remove USE_DYNAMIC_LIBRARIES from ${PETSC_DIR}/bmake/$PETSC_ARCH/petscconf.h\n and rebuild libraries before moving",libs);
  }
#else

  ierr = EPSRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  ierr = STRegisterAll(PETSC_NULL);CHKERRQ(ierr);

#endif

  /*
      Register SLEPc events
  */
  info = SlepcRegisterEvents();CHKERRQ(info);
  SlepcInitializeCalled = PETSC_TRUE;
  PetscLogInfo(0,"SlepcInitialize: SLEPc successfully started\n");
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
  PetscErrorCode ierr,info=0;
  
  PetscFunctionBegin;
  PetscLogInfo(0,"SlepcFinalize: SLEPc successfully ended!\n");

  ierr = PetscRandomDestroy(rctx);CHKERRQ(ierr);

  if (SlepcBeganPetsc) {
    info = PetscFinalize();CHKERRQ(info);
  }

  SlepcInitializeCalled = PETSC_FALSE;

  PetscFunctionReturn(info);
}

