/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/mfnimpl.h>

static PetscBool MFNPackageInitialized = PETSC_FALSE;

const char *const MFNConvergedReasons_Shifted[] = {"DIVERGED_BREAKDOWN","DIVERGED_ITS","CONVERGED_ITERATING","CONVERGED_TOL","CONVERGED_ITS","MFNConvergedReason","MFN_",0};
const char *const*MFNConvergedReasons = MFNConvergedReasons_Shifted + 2;

/*@C
  MFNFinalizePackage - This function destroys everything in the SLEPc interface
  to the MFN package. It is called from SlepcFinalize().

  Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode MFNFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&MFNList));
  CHKERRQ(PetscFunctionListDestroy(&MFNMonitorList));
  CHKERRQ(PetscFunctionListDestroy(&MFNMonitorCreateList));
  CHKERRQ(PetscFunctionListDestroy(&MFNMonitorDestroyList));
  MFNPackageInitialized       = PETSC_FALSE;
  MFNRegisterAllCalled        = PETSC_FALSE;
  MFNMonitorRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  MFNInitializePackage - This function initializes everything in the MFN package.
  It is called from PetscDLLibraryRegister() when using dynamic libraries, and
  on the first call to MFNCreate() when using static libraries.

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode MFNInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscClassId   classids[1];

  PetscFunctionBegin;
  if (MFNPackageInitialized) PetscFunctionReturn(0);
  MFNPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Matrix Function",&MFN_CLASSID));
  /* Register Constructors */
  CHKERRQ(MFNRegisterAll());
  /* Register Monitors */
  CHKERRQ(MFNMonitorRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("MFNSetUp",MFN_CLASSID,&MFN_SetUp));
  CHKERRQ(PetscLogEventRegister("MFNSolve",MFN_CLASSID,&MFN_Solve));
  /* Process Info */
  classids[0] = MFN_CLASSID;
  CHKERRQ(PetscInfoProcessClass("mfn",1,&classids[0]));
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("mfn",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventDeactivateClass(MFN_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(MFNFinalizePackage));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library
  it is in is opened.

  This one registers all the MFN methods that are in the basic SLEPc libslepcmfn
  library.
 */
SLEPC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcmfn()
{
  PetscFunctionBegin;
  CHKERRQ(MFNInitializePackage());
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
