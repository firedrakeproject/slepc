/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/svdimpl.h>       /*I "slepcsvd.h" I*/

SLEPC_EXTERN PetscErrorCode SVDCreate_Cross(SVD);
SLEPC_EXTERN PetscErrorCode SVDCreate_Cyclic(SVD);
SLEPC_EXTERN PetscErrorCode SVDCreate_LAPACK(SVD);
SLEPC_EXTERN PetscErrorCode SVDCreate_Lanczos(SVD);
SLEPC_EXTERN PetscErrorCode SVDCreate_TRLanczos(SVD);
SLEPC_EXTERN PetscErrorCode SVDCreate_Randomized(SVD);
#if defined(SLEPC_HAVE_SCALAPACK)
SLEPC_EXTERN PetscErrorCode SVDCreate_ScaLAPACK(SVD);
#endif
#if defined(SLEPC_HAVE_ELEMENTAL)
SLEPC_EXTERN PetscErrorCode SVDCreate_Elemental(SVD);
#endif
#if defined(SLEPC_HAVE_PRIMME)
SLEPC_EXTERN PetscErrorCode SVDCreate_PRIMME(SVD);
#endif

/*@C
   SVDRegisterAll - Registers all the singular value solvers in the SVD package.

   Not Collective

   Level: advanced

.seealso: SVDRegister()
@*/
PetscErrorCode SVDRegisterAll(void)
{
  PetscFunctionBegin;
  if (SVDRegisterAllCalled) PetscFunctionReturn(0);
  SVDRegisterAllCalled = PETSC_TRUE;
  CHKERRQ(SVDRegister(SVDCROSS,SVDCreate_Cross));
  CHKERRQ(SVDRegister(SVDCYCLIC,SVDCreate_Cyclic));
  CHKERRQ(SVDRegister(SVDLAPACK,SVDCreate_LAPACK));
  CHKERRQ(SVDRegister(SVDLANCZOS,SVDCreate_Lanczos));
  CHKERRQ(SVDRegister(SVDTRLANCZOS,SVDCreate_TRLanczos));
  CHKERRQ(SVDRegister(SVDRANDOMIZED,SVDCreate_Randomized));
#if defined(SLEPC_HAVE_SCALAPACK)
  CHKERRQ(SVDRegister(SVDSCALAPACK,SVDCreate_ScaLAPACK));
#endif
#if defined(SLEPC_HAVE_ELEMENTAL)
  CHKERRQ(SVDRegister(SVDELEMENTAL,SVDCreate_Elemental));
#endif
#if defined(SLEPC_HAVE_PRIMME)
  CHKERRQ(SVDRegister(SVDPRIMME,SVDCreate_PRIMME));
#endif
  PetscFunctionReturn(0);
}

/*@C
  SVDMonitorRegisterAll - Registers all the monitors in the SVD package.

  Not Collective

  Level: advanced

.seealso: SVDMonitorRegister()
@*/
PetscErrorCode SVDMonitorRegisterAll(void)
{
  PetscFunctionBegin;
  if (SVDMonitorRegisterAllCalled) PetscFunctionReturn(0);
  SVDMonitorRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(SVDMonitorRegister("first_approximation",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,SVDMonitorFirst,NULL,NULL));
  CHKERRQ(SVDMonitorRegister("first_approximation",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,SVDMonitorFirstDrawLG,SVDMonitorFirstDrawLGCreate,NULL));
  CHKERRQ(SVDMonitorRegister("all_approximations",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,SVDMonitorAll,NULL,NULL));
  CHKERRQ(SVDMonitorRegister("all_approximations",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,SVDMonitorAllDrawLG,SVDMonitorAllDrawLGCreate,NULL));
  CHKERRQ(SVDMonitorRegister("convergence_history",PETSCVIEWERASCII,PETSC_VIEWER_DEFAULT,SVDMonitorConverged,SVDMonitorConvergedCreate,SVDMonitorConvergedDestroy));
  CHKERRQ(SVDMonitorRegister("convergence_history",PETSCVIEWERDRAW,PETSC_VIEWER_DRAW_LG,SVDMonitorConvergedDrawLG,SVDMonitorConvergedDrawLGCreate,SVDMonitorConvergedDestroy));
  PetscFunctionReturn(0);
}
