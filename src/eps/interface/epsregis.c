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

.seealso: EPSRegister()
@*/
PetscErrorCode EPSRegisterAll(void)
{
  PetscFunctionBegin;
  if (EPSRegisterAllCalled) PetscFunctionReturn(0);
  EPSRegisterAllCalled = PETSC_TRUE;
  CHKERRQ(EPSRegister(EPSPOWER,EPSCreate_Power));
  CHKERRQ(EPSRegister(EPSSUBSPACE,EPSCreate_Subspace));
  CHKERRQ(EPSRegister(EPSARNOLDI,EPSCreate_Arnoldi));
  CHKERRQ(EPSRegister(EPSLANCZOS,EPSCreate_Lanczos));
  CHKERRQ(EPSRegister(EPSKRYLOVSCHUR,EPSCreate_KrylovSchur));
  CHKERRQ(EPSRegister(EPSGD,EPSCreate_GD));
  CHKERRQ(EPSRegister(EPSJD,EPSCreate_JD));
  CHKERRQ(EPSRegister(EPSRQCG,EPSCreate_RQCG));
  CHKERRQ(EPSRegister(EPSLOBPCG,EPSCreate_LOBPCG));
  CHKERRQ(EPSRegister(EPSCISS,EPSCreate_CISS));
  CHKERRQ(EPSRegister(EPSLYAPII,EPSCreate_LyapII));
  CHKERRQ(EPSRegister(EPSLAPACK,EPSCreate_LAPACK));
#if defined(SLEPC_HAVE_ARPACK)
  CHKERRQ(EPSRegister(EPSARPACK,EPSCreate_ARPACK));
#endif
#if defined(SLEPC_HAVE_TRLAN)
  CHKERRQ(EPSRegister(EPSTRLAN,EPSCreate_TRLAN));
#endif
#if defined(SLEPC_HAVE_BLOPEX)
  CHKERRQ(EPSRegister(EPSBLOPEX,EPSCreate_BLOPEX));
#endif
#if defined(SLEPC_HAVE_PRIMME)
  CHKERRQ(EPSRegister(EPSPRIMME,EPSCreate_PRIMME));
#endif
#if defined(SLEPC_HAVE_FEAST)
  CHKERRQ(EPSRegister(EPSFEAST,EPSCreate_FEAST));
#endif
#if defined(SLEPC_HAVE_SCALAPACK)
  CHKERRQ(EPSRegister(EPSSCALAPACK,EPSCreate_ScaLAPACK));
#endif
#if defined(SLEPC_HAVE_ELPA)
  CHKERRQ(EPSRegister(EPSELPA,EPSCreate_ELPA));
#endif
#if defined(SLEPC_HAVE_ELEMENTAL)
  CHKERRQ(EPSRegister(EPSELEMENTAL,EPSCreate_Elemental));
#endif
#if defined(SLEPC_HAVE_EVSL)
  CHKERRQ(EPSRegister(EPSEVSL,EPSCreate_EVSL));
#endif
  PetscFunctionReturn(0);
}

/*@C
  EPSMonitorRegisterAll - Registers all the monitors in the EPS package.

  Not Collective

  Level: advanced

.seealso: EPSMonitorRegister()
@*/
PetscErrorCode EPSMonitorRegisterAll(void)
{
  PetscFunctionBegin;
  if (EPSMonitorRegisterAllCalled) PetscFunctionReturn(0);
  EPSMonitorRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(EPSMonitorRegister("first_approximation",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,EPSMonitorFirst,NULL,NULL));
  CHKERRQ(EPSMonitorRegister("first_approximation",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,EPSMonitorFirstDrawLG,EPSMonitorFirstDrawLGCreate,NULL));
  CHKERRQ(EPSMonitorRegister("all_approximations",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,EPSMonitorAll,NULL,NULL));
  CHKERRQ(EPSMonitorRegister("all_approximations",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,EPSMonitorAllDrawLG,EPSMonitorAllDrawLGCreate,NULL));
  CHKERRQ(EPSMonitorRegister("convergence_history",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,EPSMonitorConverged,EPSMonitorConvergedCreate,EPSMonitorConvergedDestroy));
  CHKERRQ(EPSMonitorRegister("convergence_history",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,EPSMonitorConvergedDrawLG,EPSMonitorConvergedDrawLGCreate,EPSMonitorConvergedDestroy));
  PetscFunctionReturn(0);
}
