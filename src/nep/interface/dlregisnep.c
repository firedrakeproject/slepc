/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/nepimpl.h>

static PetscBool NEPPackageInitialized = PETSC_FALSE;

const char *NEPErrorTypes[] = {"ABSOLUTE","RELATIVE","NEPErrorType","NEP_ERROR_",0};
const char *NEPRefineTypes[] = {"NONE","SIMPLE","MULTIPLE","NEPRefine","NEP_REFINE_",0};
const char *NEPRefineSchemes[] = {"","SCHUR","MBE","EXPLICIT","NEPRefineScheme","NEP_REFINE_SCHEME_",0};
const char *const NEPConvergedReasons_Shifted[] = {"DIVERGED_LINEAR_SOLVE","","DIVERGED_BREAKDOWN","DIVERGED_ITS","CONVERGED_ITERATING","CONVERGED_TOL","CONVERGED_USER","NEPConvergedReason","NEP_",0};
const char *const*NEPConvergedReasons = NEPConvergedReasons_Shifted + 4;

#undef __FUNCT__
#define __FUNCT__ "NEPFinalizePackage"
/*@C
   NEPFinalizePackage - This function destroys everything in the Slepc interface
   to the NEP package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode NEPFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&NEPList);CHKERRQ(ierr);
  NEPPackageInitialized = PETSC_FALSE;
  NEPRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPInitializePackage"
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
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (NEPPackageInitialized) PetscFunctionReturn(0);
  NEPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("NEP Solver",&NEP_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = NEPRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("NEPSetUp",NEP_CLASSID,&NEP_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("NEPSolve",NEP_CLASSID,&NEP_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("NEPRefine",NEP_CLASSID,&NEP_Refine);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("NEPFunctionEval",NEP_CLASSID,&NEP_FunctionEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("NEPJacobianEval",NEP_CLASSID,&NEP_JacobianEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("NEPDerivativesEval",NEP_CLASSID,&NEP_DerivativesEval);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"nep",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(NEP_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"nep",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(NEP_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(NEPFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)

#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_slepcnep"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library
  it is in is opened.

  This one registers all the NEP methods that are in the basic SLEPc libslepcnep
  library.
 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcnep()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = NEPInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */

