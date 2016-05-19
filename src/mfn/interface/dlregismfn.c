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

#include <slepc/private/mfnimpl.h>

static PetscBool MFNPackageInitialized = PETSC_FALSE;

const char *const MFNConvergedReasons_Shifted[] = {"DIVERGED_BREAKDOWN","DIVERGED_ITS","","","CONVERGED_ITERATING","","CONVERGED_TOL","CONVERGED_ITS","MFNConvergedReason","MFN_",0};
const char *const*MFNConvergedReasons = MFNConvergedReasons_Shifted + 4;

#undef __FUNCT__
#define __FUNCT__ "MFNFinalizePackage"
/*@C
  MFNFinalizePackage - This function destroys everything in the SLEPc interface
  to the MFN package. It is called from SlepcFinalize().

  Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode MFNFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&MFNList);CHKERRQ(ierr);
  MFNPackageInitialized = PETSC_FALSE;
  MFNRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNInitializePackage"
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
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MFNPackageInitialized) PetscFunctionReturn(0);
  MFNPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Matrix Function",&MFN_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = MFNRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("MFNSetUp",MFN_CLASSID,&MFN_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MFNSolve",MFN_CLASSID,&MFN_Solve);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"mfn",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(MFN_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"mfn",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(MFN_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(MFNFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)

#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_slepcmfn"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library
  it is in is opened.

  This one registers all the MFN methods that are in the basic SLEPc libslepcmfn
  library.
 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcmfn()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MFNInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */

