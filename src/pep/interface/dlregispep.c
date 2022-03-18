/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/pepimpl.h>

static PetscBool PEPPackageInitialized = PETSC_FALSE;

const char *PEPBasisTypes[] = {"MONOMIAL","CHEBYSHEV1","CHEBYSHEV2","LEGENDRE","LAGUERRE","HERMITE","PEPBasis","PEP_BASIS_",0};
const char *PEPScaleTypes[] = {"NONE","SCALAR","DIAGONAL","BOTH","PEPScale","PEP_SCALE_",0};
const char *PEPRefineTypes[] = {"NONE","SIMPLE","MULTIPLE","PEPRefine","PEP_REFINE_",0};
const char *PEPRefineSchemes[] = {"","SCHUR","MBE","EXPLICIT","PEPRefineScheme","PEP_REFINE_SCHEME_",0};
const char *PEPExtractTypes[] = {"","NONE","NORM","RESIDUAL","STRUCTURED","PEPExtract","PEP_EXTRACT_",0};
const char *PEPErrorTypes[] = {"ABSOLUTE","RELATIVE","BACKWARD","PEPErrorType","PEP_ERROR_",0};
const char *const PEPConvergedReasons_Shifted[] = {"","DIVERGED_SYMMETRY_LOST","DIVERGED_BREAKDOWN","DIVERGED_ITS","CONVERGED_ITERATING","CONVERGED_TOL","CONVERGED_USER","PEPConvergedReason","PEP_",0};
const char *const*PEPConvergedReasons = PEPConvergedReasons_Shifted + 4;
const char *PEPJDProjectionTypes[] = {"HARMONIC","ORTHOGONAL","PEPJDProjection","PEP_JD_PROJECTION_",0};
const char *PEPCISSExtractions[] = {"RITZ","HANKEL","CAA","PEPCISSExtraction","PEP_CISS_EXTRACTION_",0};

/*@C
   PEPFinalizePackage - This function destroys everything in the Slepc interface
   to the PEP package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode PEPFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&PEPList));
  CHKERRQ(PetscFunctionListDestroy(&PEPMonitorList));
  CHKERRQ(PetscFunctionListDestroy(&PEPMonitorCreateList));
  CHKERRQ(PetscFunctionListDestroy(&PEPMonitorDestroyList));
  PEPPackageInitialized       = PETSC_FALSE;
  PEPRegisterAllCalled        = PETSC_FALSE;
  PEPMonitorRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   PEPInitializePackage - This function initializes everything in the PEP package.
   It is called from PetscDLLibraryRegister() when using dynamic libraries, and
   on the first call to PEPCreate() when using static libraries.

   Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode PEPInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscClassId   classids[1];

  PetscFunctionBegin;
  if (PEPPackageInitialized) PetscFunctionReturn(0);
  PEPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("PEP Solver",&PEP_CLASSID));
  /* Register Constructors */
  CHKERRQ(PEPRegisterAll());
  /* Register Monitors */
  CHKERRQ(PEPMonitorRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("PEPSetUp",PEP_CLASSID,&PEP_SetUp));
  CHKERRQ(PetscLogEventRegister("PEPSolve",PEP_CLASSID,&PEP_Solve));
  CHKERRQ(PetscLogEventRegister("PEPRefine",PEP_CLASSID,&PEP_Refine));
  CHKERRQ(PetscLogEventRegister("PEPCISS_SVD",PEP_CLASSID,&PEP_CISS_SVD));
  /* Process Info */
  classids[0] = PEP_CLASSID;
  CHKERRQ(PetscInfoProcessClass("pep",1,&classids[0]));
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("pep",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventDeactivateClass(PEP_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(PEPFinalizePackage));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library
  it is in is opened.

  This one registers all the PEP methods that are in the basic SLEPc libslepcpep
  library.
 */
SLEPC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcpep()
{
  PetscFunctionBegin;
  CHKERRQ(PEPInitializePackage());
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
