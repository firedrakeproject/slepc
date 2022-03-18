/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/pepimpl.h>      /*I "slepcpep.h" I*/

SLEPC_EXTERN PetscErrorCode PEPCreate_Linear(PEP);
SLEPC_EXTERN PetscErrorCode PEPCreate_QArnoldi(PEP);
SLEPC_EXTERN PetscErrorCode PEPCreate_TOAR(PEP);
SLEPC_EXTERN PetscErrorCode PEPCreate_STOAR(PEP);
SLEPC_EXTERN PetscErrorCode PEPCreate_JD(PEP);
#if defined(PETSC_USE_COMPLEX)
SLEPC_EXTERN PetscErrorCode PEPCreate_CISS(PEP);
#endif

/*@C
   PEPRegisterAll - Registers all the solvers in the PEP package.

   Not Collective

   Level: advanced

.seealso: PEPRegister()
@*/
PetscErrorCode PEPRegisterAll(void)
{
  PetscFunctionBegin;
  if (PEPRegisterAllCalled) PetscFunctionReturn(0);
  PEPRegisterAllCalled = PETSC_TRUE;
  CHKERRQ(PEPRegister(PEPLINEAR,PEPCreate_Linear));
  CHKERRQ(PEPRegister(PEPQARNOLDI,PEPCreate_QArnoldi));
  CHKERRQ(PEPRegister(PEPTOAR,PEPCreate_TOAR));
  CHKERRQ(PEPRegister(PEPSTOAR,PEPCreate_STOAR));
  CHKERRQ(PEPRegister(PEPJD,PEPCreate_JD));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PEPRegister(PEPCISS,PEPCreate_CISS));
#endif
  PetscFunctionReturn(0);
}

/*@C
  PEPMonitorRegisterAll - Registers all the monitors in the PEP package.

  Not Collective

  Level: advanced

.seealso: PEPMonitorRegister()
@*/
PetscErrorCode PEPMonitorRegisterAll(void)
{
  PetscFunctionBegin;
  if (PEPMonitorRegisterAllCalled) PetscFunctionReturn(0);
  PEPMonitorRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(PEPMonitorRegister("first_approximation",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,PEPMonitorFirst,NULL,NULL));
  CHKERRQ(PEPMonitorRegister("first_approximation",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,PEPMonitorFirstDrawLG,PEPMonitorFirstDrawLGCreate,NULL));
  CHKERRQ(PEPMonitorRegister("all_approximations",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,PEPMonitorAll,NULL,NULL));
  CHKERRQ(PEPMonitorRegister("all_approximations",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,PEPMonitorAllDrawLG,PEPMonitorAllDrawLGCreate,NULL));
  CHKERRQ(PEPMonitorRegister("convergence_history",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,PEPMonitorConverged,PEPMonitorConvergedCreate,PEPMonitorConvergedDestroy));
  CHKERRQ(PEPMonitorRegister("convergence_history",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,PEPMonitorConvergedDrawLG,PEPMonitorConvergedDrawLGCreate,PEPMonitorConvergedDestroy));
  PetscFunctionReturn(0);
}
