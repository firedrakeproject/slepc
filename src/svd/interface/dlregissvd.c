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

#include <slepc/private/svdimpl.h>

static PetscBool SVDPackageInitialized = PETSC_FALSE;

const char *SVDErrorTypes[] = {"ABSOLUTE","RELATIVE","SVDErrorType","SVD_ERROR_",0};
const char *const SVDConvergedReasons_Shifted[] = {"","","DIVERGED_BREAKDOWN","DIVERGED_ITS","CONVERGED_ITERATING","CONVERGED_TOL","CONVERGED_USER","SVDConvergedReason","SVD_",0};
const char *const*SVDConvergedReasons = SVDConvergedReasons_Shifted + 4;

#undef __FUNCT__
#define __FUNCT__ "SVDFinalizePackage"
/*@C
   SVDFinalizePackage - This function destroys everything in the Slepc interface
   to the SVD package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode SVDFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&SVDList);CHKERRQ(ierr);
  SVDPackageInitialized = PETSC_FALSE;
  SVDRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDInitializePackage"
/*@C
   SVDInitializePackage - This function initializes everything in the SVD package.
   It is called from PetscDLLibraryRegister() when using dynamic libraries, and
   on the first call to SVDCreate() when using static libraries.

   Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode SVDInitializePackage(void)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (SVDPackageInitialized) PetscFunctionReturn(0);
  SVDPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("SVD Solver",&SVD_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = SVDRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("SVDSetUp",SVD_CLASSID,&SVD_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SVDSolve",SVD_CLASSID,&SVD_Solve);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"svd",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(SVD_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"svd",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(SVD_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(SVDFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)

#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_slepcsvd"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library
  it is in is opened.

  This one registers all the SVD methods that are in the basic SLEPc libslepcsvd
  library.
 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcsvd()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SVDInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */

