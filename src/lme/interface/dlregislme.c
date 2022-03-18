/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/lmeimpl.h>

static PetscBool LMEPackageInitialized = PETSC_FALSE;

const char *LMEProblemTypes[] = {"LYAPUNOV","SYLVESTER","GEN_LYAPUNOV","GEN_SYLVESTER","DT_LYAPUNOV","STEIN","LMEProblemType","LME_",0};
const char *const LMEConvergedReasons_Shifted[] = {"DIVERGED_BREAKDOWN","DIVERGED_ITS","CONVERGED_ITERATING","CONVERGED_TOL","LMEConvergedReason","LME_",0};
const char *const*LMEConvergedReasons = LMEConvergedReasons_Shifted + 2;

/*@C
  LMEFinalizePackage - This function destroys everything in the SLEPc interface
  to the LME package. It is called from SlepcFinalize().

  Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode LMEFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&LMEList));
  CHKERRQ(PetscFunctionListDestroy(&LMEMonitorList));
  CHKERRQ(PetscFunctionListDestroy(&LMEMonitorCreateList));
  CHKERRQ(PetscFunctionListDestroy(&LMEMonitorDestroyList));
  LMEPackageInitialized       = PETSC_FALSE;
  LMERegisterAllCalled        = PETSC_FALSE;
  LMEMonitorRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  LMEInitializePackage - This function initializes everything in the LME package.
  It is called from PetscDLLibraryRegister() when using dynamic libraries, and
  on the first call to LMECreate() when using static libraries.

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode LMEInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscClassId   classids[1];

  PetscFunctionBegin;
  if (LMEPackageInitialized) PetscFunctionReturn(0);
  LMEPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Lin. Matrix Equation",&LME_CLASSID));
  /* Register Constructors */
  CHKERRQ(LMERegisterAll());
  /* Register Monitors */
  CHKERRQ(LMEMonitorRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("LMESetUp",LME_CLASSID,&LME_SetUp));
  CHKERRQ(PetscLogEventRegister("LMESolve",LME_CLASSID,&LME_Solve));
  CHKERRQ(PetscLogEventRegister("LMEComputeError",LME_CLASSID,&LME_ComputeError));
  /* Process Info */
  classids[0] = LME_CLASSID;
  CHKERRQ(PetscInfoProcessClass("lme",1,&classids[0]));
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("lme",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventDeactivateClass(LME_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(LMEFinalizePackage));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library
  it is in is opened.

  This one registers all the LME methods that are in the basic SLEPc libslepclme
  library.
 */
SLEPC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepclme()
{
  PetscFunctionBegin;
  CHKERRQ(LMEInitializePackage());
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
