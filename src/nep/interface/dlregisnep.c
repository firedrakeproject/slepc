/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/nepimpl.h>

static PetscBool NEPPackageInitialized = PETSC_FALSE;

const char *NEPErrorTypes[] = {"ABSOLUTE","RELATIVE","BACKWARD","NEPErrorType","NEP_ERROR_",0};
const char *NEPRefineTypes[] = {"NONE","SIMPLE","MULTIPLE","NEPRefine","NEP_REFINE_",0};
const char *NEPRefineSchemes[] = {"","SCHUR","MBE","EXPLICIT","NEPRefineScheme","NEP_REFINE_SCHEME_",0};
const char *NEPCISSExtractions[] = {"RITZ","HANKEL","CAA","NEPCISSExtraction","NEP_CISS_EXTRACTION_",0};
const char *const NEPConvergedReasons_Shifted[] = {"DIVERGED_SUBSPACE_EXHAUSTED","DIVERGED_LINEAR_SOLVE","","DIVERGED_BREAKDOWN","DIVERGED_ITS","CONVERGED_ITERATING","CONVERGED_TOL","CONVERGED_USER","NEPConvergedReason","NEP_",0};
const char *const*NEPConvergedReasons = NEPConvergedReasons_Shifted + 5;

/*@C
   NEPFinalizePackage - This function destroys everything in the Slepc interface
   to the NEP package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode NEPFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&NEPList));
  CHKERRQ(PetscFunctionListDestroy(&NEPMonitorList));
  CHKERRQ(PetscFunctionListDestroy(&NEPMonitorCreateList));
  CHKERRQ(PetscFunctionListDestroy(&NEPMonitorDestroyList));
  NEPPackageInitialized       = PETSC_FALSE;
  NEPRegisterAllCalled        = PETSC_FALSE;
  NEPMonitorRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   NEPInitializePackage - This function initializes everything in the NEP package.
   It is called from PetscDLLibraryRegister() when using dynamic libraries, and
   on the first call to NEPCreate() when using static libraries.

   Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode NEPInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscClassId   classids[1];

  PetscFunctionBegin;
  if (NEPPackageInitialized) PetscFunctionReturn(0);
  NEPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("NEP Solver",&NEP_CLASSID));
  /* Register Constructors */
  CHKERRQ(NEPRegisterAll());
  /* Register Monitors */
  CHKERRQ(NEPMonitorRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("NEPSetUp",NEP_CLASSID,&NEP_SetUp));
  CHKERRQ(PetscLogEventRegister("NEPSolve",NEP_CLASSID,&NEP_Solve));
  CHKERRQ(PetscLogEventRegister("NEPRefine",NEP_CLASSID,&NEP_Refine));
  CHKERRQ(PetscLogEventRegister("NEPFunctionEval",NEP_CLASSID,&NEP_FunctionEval));
  CHKERRQ(PetscLogEventRegister("NEPJacobianEval",NEP_CLASSID,&NEP_JacobianEval));
  CHKERRQ(PetscLogEventRegister("NEPResolvent",NEP_CLASSID,&NEP_Resolvent));
  CHKERRQ(PetscLogEventRegister("NEPCISS_SVD",NEP_CLASSID,&NEP_CISS_SVD));
  /* Process Info */
  classids[0] = NEP_CLASSID;
  CHKERRQ(PetscInfoProcessClass("nep",1,&classids[0]));
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("nep",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventDeactivateClass(NEP_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(NEPFinalizePackage));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library
  it is in is opened.

  This one registers all the NEP methods that are in the basic SLEPc libslepcnep
  library.
 */
SLEPC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcnep()
{
  PetscFunctionBegin;
  CHKERRQ(NEPInitializePackage());
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
