/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/mfnimpl.h>  /*I "slepcmfn.h" I*/

SLEPC_EXTERN PetscErrorCode MFNCreate_Krylov(MFN);
SLEPC_EXTERN PetscErrorCode MFNCreate_Expokit(MFN);

/*@C
  MFNRegisterAll - Registers all the matrix functions in the MFN package.

  Not Collective

  Level: advanced

.seealso:  MFNRegister()
@*/
PetscErrorCode MFNRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MFNRegisterAllCalled) PetscFunctionReturn(0);
  MFNRegisterAllCalled = PETSC_TRUE;
  ierr = MFNRegister(MFNKRYLOV,MFNCreate_Krylov);CHKERRQ(ierr);
  ierr = MFNRegister(MFNEXPOKIT,MFNCreate_Expokit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  MFNMonitorRegisterAll - Registers all the monitors in the MFN package.

  Not Collective

  Level: advanced
@*/
PetscErrorCode MFNMonitorRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MFNMonitorRegisterAllCalled) PetscFunctionReturn(0);
  MFNMonitorRegisterAllCalled = PETSC_TRUE;

  ierr = MFNMonitorRegister("error_estimate",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,MFNMonitorDefault,NULL,NULL);CHKERRQ(ierr);
  ierr = MFNMonitorRegister("error_estimate",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,MFNMonitorDefaultDrawLG,MFNMonitorDefaultDrawLGCreate,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

