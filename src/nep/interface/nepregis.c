/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/nepimpl.h>      /*I "slepcnep.h" I*/

SLEPC_EXTERN PetscErrorCode NEPCreate_RII(NEP);
SLEPC_EXTERN PetscErrorCode NEPCreate_SLP(NEP);
SLEPC_EXTERN PetscErrorCode NEPCreate_NArnoldi(NEP);
SLEPC_EXTERN PetscErrorCode NEPCreate_Interpol(NEP);
#if defined(PETSC_USE_COMPLEX)
SLEPC_EXTERN PetscErrorCode NEPCreate_CISS(NEP);
#endif
SLEPC_EXTERN PetscErrorCode NEPCreate_NLEIGS(NEP);

/*@C
   NEPRegisterAll - Registers all the solvers in the NEP package.

   Not Collective

   Level: advanced

.seealso:  NEPRegister()
@*/
PetscErrorCode NEPRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (NEPRegisterAllCalled) PetscFunctionReturn(0);
  NEPRegisterAllCalled = PETSC_TRUE;
  ierr = NEPRegister(NEPRII,NEPCreate_RII);CHKERRQ(ierr);
  ierr = NEPRegister(NEPSLP,NEPCreate_SLP);CHKERRQ(ierr);
  ierr = NEPRegister(NEPNARNOLDI,NEPCreate_NArnoldi);CHKERRQ(ierr);
  ierr = NEPRegister(NEPINTERPOL,NEPCreate_Interpol);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = NEPRegister(NEPCISS,NEPCreate_CISS);CHKERRQ(ierr);
#endif
  ierr = NEPRegister(NEPNLEIGS,NEPCreate_NLEIGS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  NEPMonitorRegisterAll - Registers all the monitors in the NEP package.

  Not Collective

  Level: advanced
@*/
PetscErrorCode NEPMonitorRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (NEPMonitorRegisterAllCalled) PetscFunctionReturn(0);
  NEPMonitorRegisterAllCalled = PETSC_TRUE;

  ierr = NEPMonitorRegister("first_approximation",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,NEPMonitorFirst,NULL,NULL);CHKERRQ(ierr);
  ierr = NEPMonitorRegister("first_approximation",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,NEPMonitorFirstDrawLG,NEPMonitorFirstDrawLGCreate,NULL);CHKERRQ(ierr);
  ierr = NEPMonitorRegister("all_approximations",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,NEPMonitorAll,NULL,NULL);CHKERRQ(ierr);
  ierr = NEPMonitorRegister("all_approximations",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,NEPMonitorAllDrawLG,NEPMonitorAllDrawLGCreate,NULL);CHKERRQ(ierr);
  ierr = NEPMonitorRegister("convergence_history",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,NEPMonitorConverged,NEPMonitorConvergedCreate,NEPMonitorConvergedDestroy);CHKERRQ(ierr);
  ierr = NEPMonitorRegister("convergence_history",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,NEPMonitorConvergedDrawLG,NEPMonitorConvergedDrawLGCreate,NEPMonitorConvergedDestroy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

