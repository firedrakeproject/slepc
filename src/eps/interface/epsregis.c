/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/epsimpl.h>  /*I "slepceps.h" I*/

SLEPC_EXTERN PetscErrorCode EPSCreate_Power(EPS);
SLEPC_EXTERN PetscErrorCode EPSCreate_Subspace(EPS);
SLEPC_EXTERN PetscErrorCode EPSCreate_Arnoldi(EPS);
SLEPC_EXTERN PetscErrorCode EPSCreate_Lanczos(EPS);
SLEPC_EXTERN PetscErrorCode EPSCreate_KrylovSchur(EPS);
SLEPC_EXTERN PetscErrorCode EPSCreate_GD(EPS);
SLEPC_EXTERN PetscErrorCode EPSCreate_JD(EPS);
SLEPC_EXTERN PetscErrorCode EPSCreate_RQCG(EPS);
SLEPC_EXTERN PetscErrorCode EPSCreate_LOBPCG(EPS);
SLEPC_EXTERN PetscErrorCode EPSCreate_CISS(EPS);
SLEPC_EXTERN PetscErrorCode EPSCreate_LyapII(EPS);
SLEPC_EXTERN PetscErrorCode EPSCreate_LAPACK(EPS);
#if defined(SLEPC_HAVE_ARPACK)
SLEPC_EXTERN PetscErrorCode EPSCreate_ARPACK(EPS);
#endif
#if defined(SLEPC_HAVE_TRLAN)
SLEPC_EXTERN PetscErrorCode EPSCreate_TRLAN(EPS);
#endif
#if defined(SLEPC_HAVE_BLOPEX)
SLEPC_EXTERN PetscErrorCode EPSCreate_BLOPEX(EPS);
#endif
#if defined(SLEPC_HAVE_PRIMME)
SLEPC_EXTERN PetscErrorCode EPSCreate_PRIMME(EPS);
#endif
#if defined(SLEPC_HAVE_FEAST)
SLEPC_EXTERN PetscErrorCode EPSCreate_FEAST(EPS);
#endif
#if defined(SLEPC_HAVE_SCALAPACK)
SLEPC_EXTERN PetscErrorCode EPSCreate_ScaLAPACK(EPS);
#endif
#if defined(SLEPC_HAVE_ELPA)
SLEPC_EXTERN PetscErrorCode EPSCreate_ELPA(EPS);
#endif
#if defined(SLEPC_HAVE_ELEMENTAL)
SLEPC_EXTERN PetscErrorCode EPSCreate_Elemental(EPS);
#endif
#if defined(SLEPC_HAVE_EVSL)
SLEPC_EXTERN PetscErrorCode EPSCreate_EVSL(EPS);
#endif

/*@C
  EPSRegisterAll - Registers all the eigenvalue solvers in the EPS package.

  Not Collective

  Level: advanced

.seealso:  EPSRegister()
@*/
PetscErrorCode EPSRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (EPSRegisterAllCalled) PetscFunctionReturn(0);
  EPSRegisterAllCalled = PETSC_TRUE;
  ierr = EPSRegister(EPSPOWER,EPSCreate_Power);CHKERRQ(ierr);
  ierr = EPSRegister(EPSSUBSPACE,EPSCreate_Subspace);CHKERRQ(ierr);
  ierr = EPSRegister(EPSARNOLDI,EPSCreate_Arnoldi);CHKERRQ(ierr);
  ierr = EPSRegister(EPSLANCZOS,EPSCreate_Lanczos);CHKERRQ(ierr);
  ierr = EPSRegister(EPSKRYLOVSCHUR,EPSCreate_KrylovSchur);CHKERRQ(ierr);
  ierr = EPSRegister(EPSGD,EPSCreate_GD);CHKERRQ(ierr);
  ierr = EPSRegister(EPSJD,EPSCreate_JD);CHKERRQ(ierr);
  ierr = EPSRegister(EPSRQCG,EPSCreate_RQCG);CHKERRQ(ierr);
  ierr = EPSRegister(EPSLOBPCG,EPSCreate_LOBPCG);CHKERRQ(ierr);
  ierr = EPSRegister(EPSCISS,EPSCreate_CISS);CHKERRQ(ierr);
  ierr = EPSRegister(EPSLYAPII,EPSCreate_LyapII);CHKERRQ(ierr);
  ierr = EPSRegister(EPSLAPACK,EPSCreate_LAPACK);CHKERRQ(ierr);
#if defined(SLEPC_HAVE_ARPACK)
  ierr = EPSRegister(EPSARPACK,EPSCreate_ARPACK);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_TRLAN)
  ierr = EPSRegister(EPSTRLAN,EPSCreate_TRLAN);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_BLOPEX)
  ierr = EPSRegister(EPSBLOPEX,EPSCreate_BLOPEX);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_PRIMME)
  ierr = EPSRegister(EPSPRIMME,EPSCreate_PRIMME);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_FEAST)
  ierr = EPSRegister(EPSFEAST,EPSCreate_FEAST);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_SCALAPACK)
  ierr = EPSRegister(EPSSCALAPACK,EPSCreate_ScaLAPACK);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_ELPA)
  ierr = EPSRegister(EPSELPA,EPSCreate_ELPA);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_ELEMENTAL)
  ierr = EPSRegister(EPSELEMENTAL,EPSCreate_Elemental);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_EVSL)
  ierr = EPSRegister(EPSEVSL,EPSCreate_EVSL);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@C
  EPSMonitorRegisterAll - Registers all the monitors in the EPS package.

  Not Collective

  Level: advanced
@*/
PetscErrorCode EPSMonitorRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (EPSMonitorRegisterAllCalled) PetscFunctionReturn(0);
  EPSMonitorRegisterAllCalled = PETSC_TRUE;

  ierr = EPSMonitorRegister("first_approximation",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,EPSMonitorFirst,NULL,NULL);CHKERRQ(ierr);
  ierr = EPSMonitorRegister("first_approximation",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,EPSMonitorFirstDrawLG,EPSMonitorFirstDrawLGCreate,NULL);CHKERRQ(ierr);
  ierr = EPSMonitorRegister("all_approximations",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,EPSMonitorAll,NULL,NULL);CHKERRQ(ierr);
  ierr = EPSMonitorRegister("all_approximations",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,EPSMonitorAllDrawLG,EPSMonitorAllDrawLGCreate,NULL);CHKERRQ(ierr);
  ierr = EPSMonitorRegister("convergence_history",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,EPSMonitorConverged,EPSMonitorConvergedCreate,EPSMonitorConvergedDestroy);CHKERRQ(ierr);
  ierr = EPSMonitorRegister("convergence_history",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,EPSMonitorConvergedDrawLG,EPSMonitorConvergedDrawLGCreate,EPSMonitorConvergedDestroy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

