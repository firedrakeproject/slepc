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

.seealso:  PEPRegister()
@*/
PetscErrorCode PEPRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PEPRegisterAllCalled) PetscFunctionReturn(0);
  PEPRegisterAllCalled = PETSC_TRUE;
  ierr = PEPRegister(PEPLINEAR,PEPCreate_Linear);CHKERRQ(ierr);
  ierr = PEPRegister(PEPQARNOLDI,PEPCreate_QArnoldi);CHKERRQ(ierr);
  ierr = PEPRegister(PEPTOAR,PEPCreate_TOAR);CHKERRQ(ierr);
  ierr = PEPRegister(PEPSTOAR,PEPCreate_STOAR);CHKERRQ(ierr);
  ierr = PEPRegister(PEPJD,PEPCreate_JD);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PEPRegister(PEPCISS,PEPCreate_CISS);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@C
  PEPMonitorRegisterAll - Registers all the monitors in the PEP package.

  Not Collective

  Level: advanced
@*/
PetscErrorCode PEPMonitorRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PEPMonitorRegisterAllCalled) PetscFunctionReturn(0);
  PEPMonitorRegisterAllCalled = PETSC_TRUE;

  ierr = PEPMonitorRegister("first_approximation",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,PEPMonitorFirst,NULL,NULL);CHKERRQ(ierr);
  ierr = PEPMonitorRegister("first_approximation",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,PEPMonitorFirstDrawLG,PEPMonitorFirstDrawLGCreate,NULL);CHKERRQ(ierr);
  ierr = PEPMonitorRegister("all_approximations",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,PEPMonitorAll,NULL,NULL);CHKERRQ(ierr);
  ierr = PEPMonitorRegister("all_approximations",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,PEPMonitorAllDrawLG,PEPMonitorAllDrawLGCreate,NULL);CHKERRQ(ierr);
  ierr = PEPMonitorRegister("convergence_history",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,PEPMonitorConverged,PEPMonitorConvergedCreate,PEPMonitorConvergedDestroy);CHKERRQ(ierr);
  ierr = PEPMonitorRegister("convergence_history",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,PEPMonitorConvergedDrawLG,PEPMonitorConvergedDrawLGCreate,PEPMonitorConvergedDestroy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

