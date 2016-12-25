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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&LMEList);CHKERRQ(ierr);
  LMEPackageInitialized = PETSC_FALSE;
  LMERegisterAllCalled  = PETSC_FALSE;
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
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (LMEPackageInitialized) PetscFunctionReturn(0);
  LMEPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Lin. Matrix Equation",&LME_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = LMERegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("LMESetUp",LME_CLASSID,&LME_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("LMESolve",LME_CLASSID,&LME_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("LMEComputeError",LME_CLASSID,&LME_ComputeError);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"lme",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(LME_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"lme",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(LME_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(LMEFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library
  it is in is opened.

  This one registers all the LME methods that are in the basic SLEPc libslepclme
  library.
 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepclme()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = LMEInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */

