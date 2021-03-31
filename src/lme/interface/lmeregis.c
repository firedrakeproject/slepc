/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/lmeimpl.h>  /*I "slepclme.h" I*/

SLEPC_EXTERN PetscErrorCode LMECreate_Krylov(LME);

/*@C
  LMERegisterAll - Registers all the matrix functions in the LME package.

  Not Collective

  Level: advanced

.seealso:  LMERegister()
@*/
PetscErrorCode LMERegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (LMERegisterAllCalled) PetscFunctionReturn(0);
  LMERegisterAllCalled = PETSC_TRUE;
  ierr = LMERegister(LMEKRYLOV,LMECreate_Krylov);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  LMEMonitorRegisterAll - Registers all the monitors in the LME package.

  Not Collective

  Level: advanced
@*/
PetscErrorCode LMEMonitorRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (LMEMonitorRegisterAllCalled) PetscFunctionReturn(0);
  LMEMonitorRegisterAllCalled = PETSC_TRUE;

  ierr = LMEMonitorRegister("error_estimate",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,LMEMonitorDefault,NULL,NULL);CHKERRQ(ierr);
  ierr = LMEMonitorRegister("error_estimate",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,LMEMonitorDefaultDrawLG,LMEMonitorDefaultDrawLGCreate,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

